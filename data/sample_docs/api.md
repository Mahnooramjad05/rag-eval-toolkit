# Northwind Analytics: API

The REST API is available on the Team and Enterprise plans. Requests are authenticated with an API key sent in the Authorization header as a bearer token.

The API allows 600 requests per minute per workspace. Requests over the limit receive HTTP status 429 with a Retry-After header.

API keys are created by workspace administrators in the settings page. Keys can be scoped to read-only access and can be revoked at any time.

Webhooks can notify an external URL when a dashboard alert fires. Each webhook request is signed with an HMAC-SHA256 signature in the X-Northwind-Signature header.

Data exports through the API return CSV or JSON. A single export can contain at most 1 million rows.
