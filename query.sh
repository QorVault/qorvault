#!/usr/bin/env bash
# Interactive query tool for BoardDocs RAG API
# Prefix question with /deep to enable query rewriting & decomposition

API_URL="http://localhost:8000/api/v1/query"

trap 'printf "\n"; exit 0' INT

while true; do
    printf "\n\033[1mQuestion (/deep for expanded search):\033[0m "
    read -r question
    [ -z "$question" ] && continue
    [ "$question" = "quit" ] && exit 0

    # Check for /deep prefix
    rewrite=false
    if [[ "$question" == /deep\ * ]]; then
        rewrite=true
        question="${question#/deep }"
        printf "\nDeep search: rewriting query + sub-queries...\n\n"
    else
        printf "\nSearching...\n\n"
    fi

    # Build JSON payload safely via Python to handle all special characters
    payload=$(python3 -c "import json,sys; print(json.dumps({'query': sys.argv[1], 'tenant_id': 'kent_sd', 'rewrite_query': sys.argv[2] == 'true'}))" "$question" "$rewrite")

    curl -s -X POST "$API_URL" \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        --max-time 120 \
    | python3 -c "
import sys, json

try:
    data = json.load(sys.stdin)
except Exception as e:
    print(f'Error parsing response: {e}')
    sys.exit(1)

if 'detail' in data:
    print(f'API error: {data[\"detail\"]}')
    sys.exit(1)

routing = data.get('routing_decision')
if routing:
    route = routing.get('route', '?')
    confidence = routing.get('confidence', 0)
    reasoning = routing.get('reasoning', '')
    print(f'Route: {route} (confidence: {confidence:.0%})')
    if reasoning:
        print(f'Reasoning: {reasoning}')
    print()

sub_queries = data.get('sub_queries')
if sub_queries:
    print(f'Sub-queries ({len(sub_queries)}):')
    for sq in sub_queries:
        print(f'  - {sq}')
    print()

print(data.get('answer', 'No answer returned.'))

citations = data.get('citations', [])
if citations:
    print()
    print('--- Sources ---')
    for c in citations:
        n = c.get('source_number', '?')
        title = c.get('title') or 'Untitled'
        committee = c.get('committee_name') or ''
        date = c.get('meeting_date') or ''
        parts = [p for p in [committee, date] if p]
        suffix = ' — ' + ', '.join(parts) if parts else ''
        print(f'[{n}] {title}{suffix}')
"
done
