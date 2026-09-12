"""Optional capabilities. Each module is independently togglable.

Contract for a feature module:
  - expose estimate(...) that never raises to the camera loop
  - return a dataclass or None
  - do not import cloud / DB / auth
  - lazy-load any heavy model
"""
