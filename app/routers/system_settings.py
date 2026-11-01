"""
System settings that an admin can change while the system is running.

GET  /api/v1/settings/public        -> no login needed (the home page calls this)
PUT  /api/v1/settings/landing-page  -> admin only, turns the public home page on/off
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.database import get_database
from app.core.dependencies import get_current_user
from app.config import settings

router = APIRouter(prefix="/settings", tags=["System settings"])

COLLECTION = "system_settings"
SETTINGS_ID = "site"

def _default_enabled() -> bool:
    """Used until an admin saves a value. Set LANDING_PAGE_ENABLED=false in .env to start with it off."""
    return settings.landing_page_enabled


async def _landing_page_enabled() -> bool:
    db = get_database()
    doc = await db[COLLECTION].find_one({"_id": SETTINGS_ID})
    if doc and "landing_page_enabled" in doc:
        return bool(doc["landing_page_enabled"])
    return _default_enabled()


def _get(user, field):
    """Works whether get_current_user returns a dict or an object."""
    return user.get(field) if isinstance(user, dict) else getattr(user, field, None)


async def require_admin(user=Depends(get_current_user)):
    if str(_get(user, "role") or "").lower() != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Only admins can change system settings.")
    return user


class LandingPageUpdate(BaseModel):
    enabled: bool


@router.get("/public")
async def get_public_settings():
    """Settings the frontend needs before anyone logs in."""
    return {"landing_page_enabled": await _landing_page_enabled()}


@router.put("/landing-page")
async def set_landing_page(body: LandingPageUpdate, admin=Depends(require_admin)):
    db = get_database()
    await db[COLLECTION].update_one(
        {"_id": SETTINGS_ID},
        {"$set": {
            "landing_page_enabled": body.enabled,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": str(_get(admin, "_id") or _get(admin, "id")),
        }},
        upsert=True,
    )
    print(f"⚙️  Home page turned {'ON' if body.enabled else 'OFF'}")
    return {"landing_page_enabled": body.enabled}