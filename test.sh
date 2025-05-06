# curl -X POST \
#   http://localhost:5001/rag_query \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "query": "When is the deadline?. Give me tips for the submission",
#     "model": "llama3.2"
#   }' | jq

# curl http://localhost:11434/api/show -d '{
#   "model": "llama3.2"
# }'


# curl -X POST http://localhost:5001/api/show \
#      -H "Content-Type: application/json" \
#      -d '{"model": "llama3.2"}'

# curl http://localhost:5001/api/tags

curl -X POST http://localhost:5001/api/chat \
     -H "Content-Type: application/json" \
     -d '{
        "model": "llama3.2",
        "stream" : false,
        "messages": [
            {
            "role": "user",
            "content": "When is the deadline?, Give only the date."
            }
        ]
    }' | jq

# curl http://localhost:11434/api/tags

# curl http://localhost:5001/api/ps


echo ""
echo ""
echo ""