# Incident Response Runbook

## Severity levels

- **SEV1**: Full outage or data loss affecting all customers. Page the
  on-call lead immediately and open a incident channel within 5 minutes.
- **SEV2**: Significant degradation affecting a subset of customers, or a
  full outage of a non-critical feature. Page on-call; incident channel
  within 15 minutes.
- **SEV3**: Minor issue with a workaround available, or degradation
  affecting internal tools only. No page required; log it and address
  during business hours.

## On-call rotation

On-call rotates weekly and is tracked in the on-call calendar. The primary
on-call engineer is paged first; if there's no acknowledgment within 5
minutes, the secondary is paged automatically. Swapping shifts requires
updating the calendar at least 24 hours in advance except for emergencies.

## During an incident

1. Acknowledge the page.
2. Open an incident channel and post an initial status within the SLA for
   the severity level.
3. Assign an incident commander if the incident is SEV1 or SEV2. The
   incident commander coordinates but does not necessarily do the hands-on
   debugging themselves.
4. Post status updates at least every 30 minutes for SEV1, every hour for
   SEV2, until resolved.
5. Once resolved, write a postmortem within 3 business days. Postmortems
   are blameless and focus on system and process gaps, not individual
   error.

## Postmortem requirements

Every SEV1 and SEV2 incident requires a postmortem doc with: a timeline, a
root cause analysis, customer impact estimate, and a list of concrete
follow-up action items each with an owner and a due date. SEV3 incidents
only require a postmortem if they recurred more than twice in a month.
