from app.core.config import config
import httpx
import asyncio

async def refresh_qontak_token() -> None:
    """
    Uses the existing REFRESH_TOKEN to get a new ACCESS_TOKEN, and updates environment variables.
    (In production, you'd save tokens back to a secure store.)
    """
    # Check if required config values are present
    if not config.QONTAK_REFRESH_TOKEN:
        print("❌ Error: QONTAK_REFRESH_TOKEN is not set in configuration")
        return
    
    if not config.QONTAK_API_BASE:
        print("❌ Error: QONTAK_API_BASE is not set in configuration")
        return
    
    print(f"🔄 Using API Base URL: {config.QONTAK_API_BASE}")
    print(f"🔄 Refresh Token (first 10 chars): {config.QONTAK_REFRESH_TOKEN[:10]}...")
    
    # OAuth token endpoint (this one showed promise with 422 error indicating it exists)
    oauth_endpoint = f"{config.QONTAK_API_BASE}/oauth/token"
    
    # Try OAuth flow first (standard approach)
    oauth_payload = {
        "grant_type": "refresh_token",
        "refresh_token": config.QONTAK_REFRESH_TOKEN
    }
    
    # Add client credentials if available
    if config.QONTAK_INTEGRATION_ID:
        oauth_payload["client_id"] = config.QONTAK_INTEGRATION_ID
        print(f"🔑 Using Integration ID as client_id: {config.QONTAK_INTEGRATION_ID[:10]}...")
    
    if config.QONTAK_CLIENT_SECRET:
        oauth_payload["client_secret"] = config.QONTAK_CLIENT_SECRET
        print(f"🔐 Using Client Secret (first 10 chars): {config.QONTAK_CLIENT_SECRET[:10]}...")
    else:
        print("⚠️ No QONTAK_CLIENT_SECRET found in configuration")
    
    # For now, let's try without client_secret first since it might not be required
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"🔗 Trying OAuth endpoint: {oauth_endpoint}")
            print(f"📦 Payload: {oauth_payload}")
            
            resp = await client.post(
                oauth_endpoint, 
                json=oauth_payload, 
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            
            print(f"📡 Response Status: {resp.status_code}")
            print(f"📡 Response Headers: {dict(resp.headers)}")
            
            if resp.status_code == 200:
                data = resp.json()
                new_access = data.get("access_token")
                new_refresh = data.get("refresh_token")
                
                if new_access:
                    print("✅ Success! Token refresh completed")
                    print(f"🔑 New Access Token (first 20 chars): {new_access[:20]}...")
                    if new_refresh:
                        print(f"🔄 New Refresh Token (first 20 chars): {new_refresh[:20]}...")
                    return
                else:
                    print("⚠️ No access_token in response")
                    print(f"📄 Response body: {resp.text}")
            else:
                print(f"❌ HTTP {resp.status_code}: {resp.text}")
                
                # If we get 422, let's try with form data instead of JSON
                if resp.status_code == 422:
                    print("🔄 Trying with form data instead of JSON...")
                    
                    form_resp = await client.post(
                        oauth_endpoint,
                        data=oauth_payload,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=30.0
                    )
                    
                    print(f"📡 Form Response Status: {form_resp.status_code}")
                    
                    if form_resp.status_code == 200:
                        data = form_resp.json()
                        new_access = data.get("access_token")
                        new_refresh = data.get("refresh_token")
                        
                        if new_access:
                            print("✅ Success with form data! Token refresh completed")
                            print(f"🔑 New Access Token (first 20 chars): {new_access[:20]}...")
                            if new_refresh:
                                print(f"🔄 New Refresh Token (first 20 chars): {new_refresh[:20]}...")
                            return
                    else:
                        print(f"❌ Form data also failed: HTTP {form_resp.status_code}: {form_resp.text}")
                        
        except Exception as e:
            print(f"❌ Error with OAuth endpoint: {e}")
                
        print("❌ Token refresh failed. Please check your refresh token and API configuration.")
        print("💡 Make sure you have valid QONTAK_REFRESH_TOKEN and QONTAK_INTEGRATION_ID in your .env file")

def main():
    asyncio.run(refresh_qontak_token())

if __name__ == "__main__":
    main()