# Credit-Based Token Usage System

A credit system where users have a balance that burns based on their LLM token consumption. Credits are tracked per-user in Supabase and deducted before each LLM operation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Backend                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │   Router    │───▶│CreditManager │───▶│TokenTracker │  │
│  │  (Auth)     │    │ (burn/refund)│    │(already has)│  │
│  └─────────────┘    └──────────────┘    └─────────────┘  │
│         │                  │                    │        │
│         ▼                  ▼                    ▼        │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Supabase PostgreSQL                     ││
│  │  profiles │ credit_transactions │ token_usage      ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│    Frontend     │  (sends user_id, displays balance)
└─────────────────┘
```

## Database Schema

### `profiles` table
```sql
ALTER TABLE profiles ADD COLUMN credits_balance DECIMAL(10, 2) DEFAULT 100.00;
```
New users start with 100 credits (configurable).

### `credit_transactions` table
```sql
CREATE TABLE credit_transactions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    amount DECIMAL(10, 2) NOT NULL,        -- negative for burn, positive for credit
    transaction_type TEXT NOT NULL,         -- 'burn', 'purchase', 'refund', 'signup_bonus'
    feature TEXT,                           -- 'search', 'generate_mcq', etc.
    tokens_used INTEGER,
    estimated_cost_usd NUMERIC(10, 6),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### `token_usage` table (updated)
```sql
ALTER TABLE token_usage ADD COLUMN user_id UUID REFERENCES profiles(id);
```

## Feature Credit Costs

Credit cost per feature (USD equivalent):

| Feature | Cost (credits) | Description |
|---------|----------------|-------------|
| `search` | 0.001 | Book context search |
| `generate_mcq` | 0.01 | Generate multiple choice questions |
| `generate_essay` | 0.015 | Generate essay questions |
| `summarize` | 0.005 | Book summarization |
| `mindmap` | 0.02 | Mind map generation |
| `dataset` | 0.025 | Full dataset generation |
| `agent_reasoning` | 0.005 | Agent chat reasoning |

## API Endpoints

### Credit Endpoints

#### GET `/credits/{user_id}`
Get user's current credit balance and recent transactions.

**Headers:**
```
x-api-key: <your-api-key>
```

**Response:**
```json
{
  "balance": 99.985,
  "recent_transactions": [
    {
      "id": "uuid",
      "amount": -0.005,
      "transaction_type": "burn",
      "feature": "agent_reasoning",
      "tokens_used": 1500,
      "estimated_cost_usd": 0.005,
      "created_at": "2026-06-02T12:00:00Z"
    }
  ]
}
```

#### POST `/credits/{user_id}/add`
Add credits to user account (admin operation).

**Headers:**
```
x-api-key: <your-api-key>
```

**Query Parameters:**
- `amount` (float, required) - Credits to add
- `transaction_type` (string, optional) - e.g., "purchase", "signup_bonus", "refund"

**Response:**
```json
{
  "status": "success",
  "new_balance": 150.00
}
```

#### POST `/credits/{user_id}/burn`
Manually burn credits for a user.

**Query Parameters:**
- `feature` (string, optional) - Feature name, default "unknown"
- `amount` (float, optional) - Specific amount to burn

---

### Feature Endpoints with Credit Check

All LLM-consuming endpoints now require `user_id` in the request body/query and perform credit checks.

#### POST `/book-qa/stream`

**Headers:**
```
x-api-key: <your-api-key>
```

**Request Body:**
```json
{
  "user_id": "uuid-of-user",
  "session_id": "chat-session-id",
  "book_id": "book-id",
  "messages": [
    {"role": "user", "content": "What is the main topic?"}
  ]
}
```

**Response (SSE stream):**
```
data: {"id": "chatcmpl", "type": "final", "content": "...", "metadata": {}}
data: [DONE]
```

**Error (insufficient credits):**
```
HTTP 402 Payment Required
{
  "detail": "Insufficient credits. Balance too low for feature 'agent_reasoning' (cost: 0.005)"
}
```

#### POST `/book-qa/mindmap`

**Query Parameters:**
- `user_id` (string, required)
- `book_id` (string, required)
- `user_prompt` (string, optional)

**Response:**
```json
{
  "title": "Book Title",
  "mermaid": "mindmap\n  root(...)",
  "sources": ["source1", "source2"]
}
```

#### POST `/book-qa/dataset`

**Query Parameters:**
- `user_id` (string, required)
- `book_id` (string, required)
- `difficulty` (string, optional) - "easy", "medium", "hard"
- `num_mcq` (int, optional) - Number of MCQ questions
- `num_essay` (int, optional) - Number of essay questions

#### POST `/book-qa/summarize`

**Query Parameters:**
- `user_id` (string, required)
- `book_id` (string, required)
- `user_prompt` (string, optional)

---

## Client Integration

### Frontend Request Example

```typescript
const response = await fetch('/book-qa/stream', {
  method: 'POST',
  headers: {
    'x-api-key': 'your-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_id: 'uuid-from-your-auth-system',
    session_id: 'unique-session-id',
    book_id: 'course-123',
    messages: [{ role: 'user', content: 'Explain chapter 1' }]
  })
});

if (response.status === 402) {
  // Show "Insufficient credits" to user
  const data = await response.json();
  alert('Please purchase more credits');
} else if (response.ok) {
  // Stream response
  const reader = response.body.getReader();
  // ...
}
```

### Check Balance

```typescript
async function getBalance(userId: string) {
  const response = await fetch(`/credits/${userId}`, {
    headers: { 'x-api-key': 'your-api-key' }
  });
  const data = await response.json();
  return data.balance;
}
```

---

## Credit Flow

### Normal Flow
```
1. Request arrives
2. CreditManager.check_sufficient_credits() → true
3. CreditManager.burn_credits() → deducted from balance
4. LLM processes request
5. TokenTracker records actual usage with user_id
6. Response returned
```

### Error Flow (LLM Failure)
```
1. Request arrives
2. Credits burned
3. LLM throws error
4. CreditManager.refund_credits() → credits restored
5. Error returned to client
```

---

## Configuration

### Feature Costs

Edit `core/utils/credit_manager.py`:

```python
FEATURE_COST = {
    "search": 0.001,
    "generate_mcq": 0.01,
    "generate_essay": 0.015,
    "summarize": 0.005,
    "mindmap": 0.02,
    "dataset": 0.025,
    "agent_reasoning": 0.005,
    "unknown": 0.001,  # fallback
}
```

### Default Balance

Modify the SQL migration or update directly:

```sql
UPDATE profiles SET credits_balance = 100.00 WHERE credits_balance IS NULL;
```

---

## Security Notes

- **Trust Model:** Frontend sends `user_id`. For production, replace with Supabase Auth JWT verification.
- **API Key:** Protects endpoints but doesn't identify users.
- **Atomic Transactions:** Credit burns use `FOR UPDATE` locks to prevent race conditions.
- **Audit Log:** All transactions logged in `credit_transactions` table.

### Upgrading to Real Auth (Recommended)

Replace the simple API key check with Supabase Auth JWT:

```python
from supabase import create_client

async def verify_user(request: Request) -> str:
    """Extract user_id from Supabase JWT."""
    auth_header = request.headers.get('Authorization')
    if not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = auth_header[7:]
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    user = client.auth.get_user(token)

    return user.id  # Use as user_id
```

---

## Error Codes

| HTTP Status | Meaning |
|------------|---------|
| 402 | Insufficient credits |
| 403 | Invalid/missing API key |
| 500 | Credit transaction failed (database error) |

---

## Monitoring

### Check Redis-free in-memory tracking

```python
from core.utils.credit_manager import CreditManager

cm = CreditManager()
summary = cm.get_summary()  # Not implemented yet
```

### Query Database Directly

```sql
-- User's total spent
SELECT SUM(ABS(amount)) as total_spent
FROM credit_transactions
WHERE user_id = 'uuid' AND transaction_type = 'burn';

-- Daily usage
SELECT DATE(created_at), SUM(ABS(amount))
FROM credit_transactions
WHERE user_id = 'uuid'
GROUP BY DATE(created_at);
```
