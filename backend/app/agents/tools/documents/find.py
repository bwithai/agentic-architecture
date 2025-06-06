import json
from typing import Dict, Any, Optional, List
import re

from app.mongodb.client import MongoDBClient
from app.agents.tools.base.tool import BaseTool, ToolResponse
from app.agents.utils.serialization_utils import mongodb_json_dumps, serialize_mongodb_doc


class FindTool(BaseTool):
    """Tool to query documents from a MongoDB collection."""

    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
    
    @property
    def name(self) -> str:
        return "find"
    
    @property
    def description(self) -> str:
        return "Query documents in a collection using MongoDB query syntax. For user searches, supports partial name matches and will search entire collection if needed. When searching for specific users, use limit:1000 and search_mode:'fuzzy'."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Name of the collection to query"
                },
                "filter": {
                    "type": "object",
                    "description": "MongoDB query filter",
                    "default": {}
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum documents to return. Use 1000 for specific user searches.",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 1000
                },
                "projection": {
                    "type": "object",
                    "description": "Fields to include/exclude",
                    "default": {}
                },
                "skip": {
                    "type": "number",
                    "description": "Number of documents to skip (for pagination)",
                    "default": 0,
                    "minimum": 0
                },
                "search_mode": {
                    "type": "string",
                    "description": "Search mode: 'exact' (default) or 'fuzzy' for partial matches. Use 'fuzzy' for user name searches.",
                    "default": "exact"
                }
            },
            "required": ["collection"]
        }

    def _create_fuzzy_filter(self, original_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fuzzy search filter for text fields."""
        fuzzy_filter = {}
        
        # Special handling for user name fields
        user_name_fields = ["user_name", "name", "username", "full_name"]
        
        for key, value in original_filter.items():
            if isinstance(value, str):
                if key in user_name_fields:
                    # For name fields, create an OR condition with multiple variations
                    name_variations = [
                        {"$regex": f"^{re.escape(value)}$", "$options": "i"},  # Exact match (case-insensitive)
                        {"$regex": f".*{re.escape(value)}.*", "$options": "i"},  # Contains
                        {"$regex": f"^{re.escape(value)}", "$options": "i"},  # Starts with
                        {"$regex": f"{re.escape(value)}$", "$options": "i"}  # Ends with
                    ]
                    fuzzy_filter["$or"] = [
                        {name_field: variation}
                        for name_field in user_name_fields
                        for variation in name_variations
                    ]
                else:
                    # For non-name fields, use simple contains match
                    fuzzy_filter[key] = {"$regex": f".*{re.escape(value)}.*", "$options": "i"}
            else:
                fuzzy_filter[key] = value
        return fuzzy_filter

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Query documents from a MongoDB collection with smart search capabilities.
        
        Args:
            params: Dictionary containing:
                - collection: Name of the collection
                - filter: (Optional) Query filter
                - limit: (Optional) Maximum documents to return
                - projection: (Optional) Fields to include/exclude
                - skip: (Optional) Number of documents to skip
                - search_mode: (Optional) 'exact' or 'fuzzy'
                
        Returns:
            ToolResponse with the query results
        """
        try:
            # Validate collection name
            collection = self.validate_collection(params.get("collection"))
            
            # Get optional parameters with defaults
            original_filter = params.get("filter", {})
            limit = min(params.get("limit", 10), 1000)
            projection = params.get("projection", {})
            skip = max(params.get("skip", 0), 0)
            search_mode = params.get("search_mode", "exact")
            
            # Get MongoDB db directly from the module
            if self.mongodb_client.db is None:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: MongoDB database connection is not initialized."
                    }],
                    is_error=True
                )

            # Special handling for user searches
            is_user_search = (
                collection == "users" and 
                any(key in original_filter for key in ["user_name", "name", "username", "full_name", "email"])
            )

            # For specific user searches, ensure we use a large enough limit
            if is_user_search and any(isinstance(v, str) for v in original_filter.values()):
                limit = max(limit, 1000)

            # Try exact match first
            filter_obj = original_filter
            results = list(
                self.mongodb_client.db[collection]
                .find(filter_obj, projection)
                .skip(skip)
                .limit(limit)
            )

            # If no results found and it's a user search or fuzzy mode, try fuzzy search
            if (not results) and (is_user_search or search_mode == "fuzzy"):
                fuzzy_filter = self._create_fuzzy_filter(original_filter)
                results = list(
                    self.mongodb_client.db[collection]
                    .find(fuzzy_filter, projection)
                    .skip(skip)
                    .limit(limit)
                )

            # If still no results and it's a user search, get total count
            if not results and is_user_search:
                total_count = self.mongodb_client.db[collection].count_documents({})
                if total_count > limit:
                    # Do a full collection scan in batches
                    batch_size = 1000
                    for batch_skip in range(0, total_count, batch_size):
                        batch_results = list(
                            self.mongodb_client.db[collection]
                            .find(fuzzy_filter, projection)
                            .skip(batch_skip)
                            .limit(batch_size)
                        )
                        if batch_results:
                            results = batch_results
                            break

            # Use our custom serialization utility for MongoDB types
            serialized_results = serialize_mongodb_doc(results)
            
            # Get total count for metadata
            total_count = (
                self.mongodb_client.db[collection].count_documents({})
                if is_user_search or len(results) >= limit
                else len(results)
            )
            
            # Add metadata about the search
            response_data = {
                "results": serialized_results,
                "metadata": {
                    "total_found": len(results),
                    "total_in_collection": total_count,
                    "search_mode": "fuzzy" if (not results and is_user_search) else "exact",
                    "skip": skip,
                    "limit": limit,
                    "has_more": total_count > (skip + limit)
                }
            }
            
            # Return success response
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": mongodb_json_dumps(response_data)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"Find error: {str(error)}")
            return self.handle_error(error)
