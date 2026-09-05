# API Design Guidelines

## Versioning

All public endpoints are versioned via the URL path, e.g. `/v1/orders`. We
do not use header-based versioning. A breaking change always ships as a new
version; we support the previous major version for at least 6 months after
a new one ships.

## Pagination

List endpoints use cursor-based pagination, not offset-based. Every list
response includes a `next_cursor` field, which is `null` when there are no
more results. Clients should treat the cursor as an opaque string and never
attempt to construct one manually. The default page size is 25 and the
maximum is 100.

## Error format

Errors are returned as JSON with three fields: `error_code` (a stable
machine-readable string), `message` (a human-readable description), and
`request_id` (for support lookups). HTTP status codes follow standard
semantics: 4xx for client errors, 5xx for server errors. We never return a
200 status with an error body.

## Rate limits

Each API key is limited to 100 requests per minute using a sliding-window
counter, not a fixed window. When a client exceeds the limit, the API
returns `429 Too Many Requests` with a `Retry-After` header indicating how
many seconds to wait. Rate limit counters are shared across all endpoints
for a given key, not tracked per-endpoint.

## Authentication

All requests must include an `Authorization: Bearer <token>` header. Tokens
expire after 24 hours and must be refreshed using the `/v1/auth/refresh`
endpoint. There is no support for long-lived API tokens at this time.
