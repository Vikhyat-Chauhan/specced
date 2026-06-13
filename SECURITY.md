# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in Specced, please report it **privately** — do not open a public GitHub issue, pull request, or discussion, as that exposes the issue before a fix is available.

To report:

- Use **GitHub's private vulnerability reporting** (the repository's *Security* tab →
  *Report a vulnerability*), **or**
- Email the maintainer at **vikhyat.chauhan@gmail.com** with the subject line
  `Specced security`.

Please include a description of the issue and its impact, steps to reproduce (a proof of concept if you have one), and the affected files or components.

## What to expect

- We aim to acknowledge a report within **3 business days**.
- We'll keep you updated on the assessment and remediation, and coordinate disclosure timing with you once a fix is ready.
- Please give us a reasonable window to address the issue before any public disclosure.

## Scope

In scope: the Specced code in this repository.

**Protected health information (PHI):** Specced is local-first precisely so that PHI never leaves the host. This repository must contain **no real patient data** — only synthetic data (Faker + the curated knowledge base) or public de-identified benchmarks. If you find anything resembling real PHI committed to the repo, treat it as a security issue and report it privately using the process above.

Out of scope: vulnerabilities in third-party dependencies or model providers — report those to the respective project.

## Operational hardening

How Specced is run safely on-premise (local serving, the no-real-PHI data policy, secret handling, reproducibility) is documented in [`docs/OPERATIONS.md`](docs/OPERATIONS.md).
