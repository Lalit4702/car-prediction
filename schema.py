from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class FuelType(str, Enum):
    petrol = "Petrol"
    diesel = "Diesel"
    cng = "CNG"


class SellerType(str, Enum):
    dealer = "Dealer"
    individual = "Individual"


class TransmissionType(str, Enum):
    manual = "Manual"
    automatic = "Automatic"


class CarFeatures(BaseModel):
    Car_Name: str = Field(..., example="swift")
    Year: int = Field(..., ge=1990, le=2030, example=2014)
    Present_Price: float = Field(..., ge=0.01, example=5.59, description="Current ex-showroom price in Lakhs")
    Kms_Driven: int = Field(..., ge=0, example=40000, description="Total kilometers driven")
    Fuel_Type: FuelType = Field(default=FuelType.petrol)
    Seller_Type: SellerType = Field(default=SellerType.dealer)
    Transmission: TransmissionType = Field(default=TransmissionType.manual)
    Owner: int = Field(
        ..., ge=0, le=3, example=0, description="Number of previous owners (0, 1, or 3)"
    )


class PredictionResponse(BaseModel):
    prediction_price: float = Field(..., description="Predicted selling price in Lakhs")
    inr_formatted: Optional[str] = Field(None, description="Formatted price in Indian Rupees")
    fair_price_low: Optional[float] = Field(None, description="Estimated lower bound")
    fair_price_high: Optional[float] = Field(None, description="Estimated upper bound")
    depreciation_pct: Optional[float] = Field(None, description="Percentage depreciated from present price")
    currency: Optional[str] = Field(default="Lakhs (INR)")
    status: str = Field(default="success")
