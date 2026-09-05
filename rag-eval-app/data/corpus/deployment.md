# Deploying the Service

## Environments

We run three environments: `dev`, `staging`, and `production`. Each environment
has its own configuration file under `config/<env>.yaml` and its own database.
Never point a `dev` deployment at the `production` database — the deploy
script will refuse to run if it detects a mismatched environment tag.

## Deployment process

Deployments are triggered by pushing a tag matching `v*.*.*` to the `main`
branch. The CI pipeline runs the full test suite, builds a Docker image,
pushes it to the internal registry, and then triggers a rolling deploy via
the `deploy.sh` script. A rolling deploy replaces one instance at a time and
waits for the new instance to pass its health check before moving to the
next one.

Rollbacks are done by re-running `deploy.sh` with the previous image tag.
There is no automatic rollback on failed health checks in production yet —
this is a known gap and is tracked as an open task.

## Health checks

Every service exposes a `/healthz` endpoint that returns `200 OK` when the
service can reach its database and its message queue. The load balancer
polls this endpoint every 10 seconds and removes an instance from rotation
after 3 consecutive failures.

## Secrets

Secrets are stored in the internal secrets manager, not in environment
variables or config files. Each service pulls its secrets at startup using
its service identity token. Rotating a secret requires updating it in the
secrets manager and then restarting the affected services — there is no
hot-reload of secrets.
