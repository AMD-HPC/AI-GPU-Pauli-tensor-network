# SWOSRB Jira Checklist — AI-GPU-Pauli-tensor-network Open Source Release

*Created 2026-10-09*

## Before you start
- [ ] Have Ashish's approval email ready to attach (screenshot or PDF)
- [ ] Have the GitHub repo URL ready: `github.com/AMD-HPC/AI-GPU-Pauli-tensor-network`
- [ ] Know your code scan team (SIT or AECG DevOps?)

## Create the Jira request
- [ ] Go to https://amd.atlassian.net/issues/?filter=201249
- [ ] Click **"+ Create"**
- [ ] Space: **OSRB Request (SWOSRB)**
- [ ] Work Type: **Open Sourcing Pre-Approval**

## Fill request details (Edit Request Details)
- [ ] Project name: `AI-GPU-Pauli-tensor-network`
- [ ] License: **MIT**
- [ ] Project description (see below)
- [ ] VP approval: attach Ashish's email
- [ ] Internal repo URL: `github.com/AMD-HPC/AI-GPU-Pauli-tensor-network`
- [ ] Supporting publications: arXiv preprints, QIML 2026

## Submit
- [ ] Click **"Request Approval"** transition → status becomes "Approvals Pending"
- [ ] Set calendar reminder: monitor Jira comments every day for 7 business days
- [ ] Respond to any questions within **2 business days**

## After OSRB approved
- [ ] Download OSRB approval document
- [ ] Re-submit ServiceNow: "Request for Public Repo/Change Repo Visibility to Public"
- [ ] Attach OSRB approval document

---

## Project description (for Jira ticket)

GPU-accelerated belief propagation tensor network simulator for 2D quantum lattice models.
Implements Pauli transfer matrix (PTM) operator evolution on AMD CDNA GPUs via HIP/ROCm.
Supports multi-product formula (MPF) error mitigation and Hamiltonian tomography via
operator lightcone wavemaps. Intended for quantum computing researchers studying operator
spreading, noise characterization, and error mitigation on near-term quantum hardware.
MIT license. AMD will maintain with ongoing research contributions.

---

## Status log

| Date       | Event |
|------------|-------|
| 2026-09-10 | Email sent to Sudip Nag requesting VP approval |
| 2026-09-24 | Follow-up email sent, CC: Ralph Wittig, Ashish Sirasao |
| 2026-10-09 | VP approval received from Ashish Sirasao ("I approve open sourcing") |
| 2026-10-09 | ServiceNow RITM1270874 submitted (rejected — OSRB ticket required first) |
| 2026-10-09 | SWOSRB Jira checklist created — next step: file Jira ticket |
| 2026-10-09 | SWOSRB Jira ticket filed: https://amd.atlassian.net/browse/SWOSRB-647 |
| 2026-10-09 | "Request Approval" clicked — status now "Approvals Pending" — approvers notified |

---

## Reference links

- AMD GitHub FAQ: https://amdcloud.sharepoint.com/sites/AISEPublic/SitePages/AMD-GitHub-FAQs.aspx#how-to-create-a-public-repo-or-fork
- SWOSRB Jira: https://amd.atlassian.net/issues/?filter=201249
- SWOSRB User Guide: https://amdcloud.sharepoint.com/sites/amd-legal/Shared%20Documents/Forms/AllItems.aspx?id=/sites/amd-legal/Shared%20Documents/SWOSRB_Jira/SWOSRBJira_UserGuide.pdf
- SWOSRB Requestors Guide: /home/paolod/Downloads/SWOSRBJira_RequestorsGuide.pdf
