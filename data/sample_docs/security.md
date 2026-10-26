# Northwind Analytics: security and data handling

All customer data is encrypted at rest with AES-256 and in transit with TLS 1.2 or higher.

Single sign-on is available on the Enterprise plan and supports SAML 2.0 and OpenID Connect. Identity providers such as Okta and Azure Active Directory are supported.

Raw event data is retained for 13 months. Aggregated dashboard data is retained for as long as the workspace exists.

When a workspace is deleted, all of its data is permanently removed within 30 days, including backups.

Customer data is hosted in two regions: Frankfurt for European customers and Virginia for North American customers. Customers choose the region when the workspace is created and it cannot be changed later.

Two-factor authentication can be enforced by workspace administrators on every plan.
