```bash
# Export db
docker exec kami_mongo mongodump --db=kami --out=/dump
docker cp kami_mongo:/dump ./mongo_backup

# Import db
docker cp ./mongo_backup MONGODB:/dump
docker exec MONGODB mongorestore --db=kami /dump/kami
```