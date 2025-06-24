# 🌐 API Documentation

## Overview

This service exposes **1 RESTful API endpoints** for web api service with background processing capabilities.

## Base URL

```
http://localhost
```

## Authentication


*Authentication methods not automatically detected.*

Common authentication patterns to verify:
- JWT tokens in Authorization header
- API keys in headers or query parameters  
- Session-based authentication
- OAuth 2.0 flows

Check the security assessment for authentication details.


## API Groups

### Generate-Course Operations

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/generate-course` | Triggers the course generation process using an external service. | org_id, user_id, course_id, s3_bucket, provider, model, api_key, max_tokens_per_call, max_tokens |

#### `POST /generate-course`
**Handler**: `trigger_course_generation_async`
**Description**: Triggers the course generation process using an external service.

**Parameters**:
- `org_id`
- `user_id`
- `course_id`
- `s3_bucket`
- `provider`
- `model`
- `api_key`
- `max_tokens_per_call`
- `max_tokens`

**Example Request**:
```bash
curl -X POST 'http://localhost/generate-course'
```


## Data Models

*No data models detected in the current analysis.*

## Error Handling


### Standard HTTP Status Codes

| Status Code | Meaning | Description |
|-------------|---------|-------------|
| `200` | OK | Request successful |
| `201` | Created | Resource created successfully |
| `400` | Bad Request | Invalid request parameters |
| `401` | Unauthorized | Authentication required |
| `403` | Forbidden | Access denied |
| `404` | Not Found | Resource not found |
| `500` | Internal Server Error | Server error occurred |

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details"
  }
}
```


## Rate Limiting


*Rate limiting configuration not automatically detected.*

Recommended rate limiting strategy:
- **Public endpoints**: 100 requests per minute per IP
- **Authenticated endpoints**: 1000 requests per minute per user
- **Resource-intensive operations**: 10 requests per minute per user

Monitor API usage and adjust limits based on actual traffic patterns.


## Examples


### Triggers the course generation process using an external service.

**Request**:
```bash
curl -X POST 'http://localhost/generate-course' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_TOKEN'
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "message": "Response data here"
  }
}
```


---

*🤖 Auto-generated API documentation by InfraDoc 2.0*
