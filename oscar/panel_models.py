"""Pure helpers for the composer profile/model selector."""


def catalog_models(models) -> list[str]:
    """Return unique, non-empty catalog model names in provider order."""
    if not isinstance(models, (list, tuple)):
        return []
    result = []
    seen = set()
    for value in models:
        name = str(value or "").strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def model_availability(model: str, models, error: str = "") -> str:
    """Describe catalog evidence without claiming that a listed model is ready to call."""
    model = str(model or "").strip()
    available = catalog_models(models)
    if model and model in available:
        return "Listed by provider"
    if error:
        return "Not verified — " + str(error).strip()
    if model:
        return "Not listed by provider"
    return "No model configured"


def selected_profile(profiles, name: str):
    """Find a configured profile by name; malformed or missing names return None."""
    name = str(name or "")
    return next((profile for profile in profiles if getattr(profile, "name", "") == name), None)
