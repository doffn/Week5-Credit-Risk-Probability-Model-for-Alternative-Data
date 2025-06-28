from pydantic import BaseModel

class CustomerData(BaseModel):
    customer_id: str
    transaction_count: int
    avg_transaction_amount: float
    frequency: float
    recency: int
    monetary: float
