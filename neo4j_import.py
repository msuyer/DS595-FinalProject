from neo4j import GraphDatabase
import json

# Connect to Neo4j
uri = "bolt://localhost:7687"
user = "neo4j"
password = "ds595liu"
driver = GraphDatabase.driver(uri, auth=(user, password))

# Load JSON data
with open('train_data.json', 'r') as file:
    data = json.load(file)

# Define Cypher query to create publications and relationships
query = """
MERGE (p:Publication {id: $publication_ID})
SET p.pubDate = $pubDate,
    p.language = $language,
    p.title = $title,
    p.abstract = $abstract,
    p.doi = $doi

WITH p
MERGE (j:Journal {name: $journal})
MERGE (p)-[:PUBLISHED_IN]->(j)

WITH p
UNWIND $authors AS author
WITH p, author
WHERE author.id IS NOT NULL
MERGE (a:Author {id: author.id})
SET a.name = author.name
MERGE (p)-[:HAS_AUTHOR]->(a)


WITH p, split(toString($keywords), ';') AS keywordList
UNWIND keywordList AS keyword
MERGE (k:Keyword {name: keyword})
MERGE (p)-[:HAS_KEYWORD]->(k)
"""

# Define Cypher query to establish CITES relationships
cite_query = """
MATCH (source:Publication {id: $source_id}), (target:Publication {id: $target_id})
MERGE (source)-[:CITES]->(target)
"""

# Execute Cypher queries for each publication
with driver.session() as session:
    for item in data:
        try:
            # Create or update publication nodes and related nodes
            session.run(query, **item)

            # Create CITES relationships for citations
            if 'Citations' in item:
                citations = str(item['Citations']).split(';')
                for cited_id in citations:
                    cited_id = int(cited_id)
                    session.run(cite_query, source_id=item['publication_ID'], target_id=cited_id)
        except Exception as e:
            print(f"Error processing publication {item['publication_ID']}: {e}")

# Close the Neo4j driver
driver.close()
