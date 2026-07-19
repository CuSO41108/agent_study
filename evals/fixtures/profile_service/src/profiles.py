def display_name(profile: dict) -> str:
    first = str(profile.get("first_name", "")).strip()
    last = str(profile.get("last_name", "")).strip()
    return " ".join(part for part in (first, last) if part)


def active_emails(profiles: list[dict]) -> list[str]:
    return sorted(
        str(profile["email"]).strip().lower()
        for profile in profiles
        if profile.get("active") and profile.get("email")
    )
