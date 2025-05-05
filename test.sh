curl -X POST \
  http://localhost:5001/rag_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "When is the deadline?. Give me tips for the submission",
    "model": "llama3.2"
  }' | jq

echo ""
echo ""
echo ""