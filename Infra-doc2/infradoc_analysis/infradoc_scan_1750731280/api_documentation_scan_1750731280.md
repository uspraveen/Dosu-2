# 🌐 API Documentation

## Overview
This document describes the APIs discovered in the analyzed infrastructure.

## Base URL
```
http://localhost
```

## Endpoints

### `POST /generate-course`
**Handler**: `trigger_course_generation_async`  
**Description**: Triggers the external course generation process using the specified parameters.

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


---
*Auto-generated API documentation by InfraDoc 2.0*
