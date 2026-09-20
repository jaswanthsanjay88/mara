# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

---

## Reporting a Vulnerability

Security is paramount when connecting neural models to physical hardware, GPIO pins, and network services.

If you discover a security vulnerability in Mara (such as an unauthenticated command execution risk, sandbox escape, or deserialization vulnerability):

1. **Do not open a public GitHub issue.**
2. Submit a private advisory via GitHub Security Advisories at [https://github.com/jaswanthsanjay88/mara/security/advisories](https://github.com/jaswanthsanjay88/mara/security/advisories).
3. Include:
   - A description of the vulnerability and potential impact.
   - Minimal proof-of-concept code or reproduction steps.
   - Suggested mitigations, if known.

You will receive an initial response within 48 hours, and critical fixes will be prioritized for immediate release.

---

## Safety Guidelines for Edge Deployment

- **Hardware Isolation**: When binding GPIO pins to electrical relays or high-voltage hardware, always enforce physical limits, thermal cutoffs, and safety interlocks independent of software control.
- **Permission Scoping**: Restrict tool registry dispatch using whitelist filters so that untrusted user prompts cannot invoke privileged system commands.
