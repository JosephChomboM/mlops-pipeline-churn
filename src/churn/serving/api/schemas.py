from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
YesNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    customer_id: str | None = None
    tenure: int = Field(..., ge=0, le=100, examples=[2])
    monthly_charges: float = Field(..., ge=0, le=200, examples=[89.1])
    total_charges: float | None = Field(None, ge=0, examples=[178.2])
    gender: Literal["Male", "Female"]
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: YesNoInternet
    online_backup: YesNoInternet
    device_protection: YesNoInternet
    tech_support: YesNoInternet
    streaming_tv: YesNoInternet
    streaming_movies: YesNoInternet
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float
    contact: bool
    expected_value: float
    risk_band: str
    decision_rule: str
    model_version: int
