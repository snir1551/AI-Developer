# About project
This project was created as part of ML 2024 course in Technion continuing education
Project focus on classic machine learning topics
Source of data https://www.sportmonks.com/football-api/

Data was parsed via API 2.0 and stored to mongodb to easy access

# Run project
Install dependencies from requirements.txt
create db_config.json file in the root directory with the following content:
```
{
  "connectionString": "your connectionString",
  "dbName": "your db name with data",
  "collectionName": "your collection name with data",
}
```