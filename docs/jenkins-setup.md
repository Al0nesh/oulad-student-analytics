# Running the Jenkins CI/CD pipeline locally (Windows + Docker Desktop)

This guide sets up the whole pipeline from [`Jenkinsfile`](../Jenkinsfile) on one laptop:

```
GitHub (push) ──poll──► Jenkins (container) ──► test image ──► lint ──► unit tests
                                   │                                         │
                                   └──► runtime image ──► smoke test ──► test server :8081
                                                              │
                                                              ├──► Docker Hub (optional)
                                                              └──► Minikube (optional)
```

## 0. What you need

| Tool | Why | Check |
|---|---|---|
| Docker Desktop (WSL 2 backend) | builds and runs all containers | `docker version` |
| Git | to push commits | `git --version` |
| Minikube *(optional)* | local Kubernetes for the CD stage | `minikube version` |
| Docker Hub account *(optional)* | to push the image | — |

All commands below are for **PowerShell**, run from the repository folder.

## 1. Start Jenkins

```powershell
cd jenkins
docker compose up -d --build      # first run downloads Jenkins + plugins, ~5 min
docker logs -f jenkins            # wait for "Jenkins is fully up and running", then Ctrl+C
```

Open <http://localhost:8080> and sign in as **admin / admin**
(change it with `$env:JENKINS_ADMIN_PASSWORD="..."` before `docker compose up`).

There is no setup wizard: [`jenkins/casc.yaml`](../jenkins/casc.yaml) (Configuration as Code)
creates the admin user and the pipeline job **oulad-analytics**, which reads the
`Jenkinsfile` from the `main` branch on GitHub.

## 2. First build

1. Open the job **oulad-analytics** → **Build Now**.
   The first build registers the parameters and the `pollSCM` trigger from the Jenkinsfile.
2. The first run takes longer (it downloads Python, the pinned libraries and CPU PyTorch into
   Docker's build cache); later runs reuse the cached layers.
3. When it is green, open <http://localhost:8081> — the test server started by the pipeline.

From now on, **every commit pushed to `main` is built automatically** within ~2 minutes:

```powershell
git commit -am "docs: small change" ; git push
```

## 3. (Optional) Push to Docker Hub

1. Docker Hub → Account settings → **Personal access tokens** → create a token (Read & Write).
2. Jenkins → Manage Jenkins → **Credentials** → System → Global → Add credentials:
   *Kind* = Username with password, *Username* = your Docker Hub user,
   *Password* = the token, *ID* = `dockerhub-creds`.
3. **Build with Parameters**: `IMAGE_NAME` = `<your-user>/oulad-analytics`, tick `PUSH_TO_DOCKERHUB`.

## 4. (Optional) Deploy to Kubernetes with Minikube

```powershell
minikube start --driver=docker --container-runtime=docker
```

Then **Build with Parameters** and tick `DEPLOY_TO_K8S`. The pipeline:

1. connects the Jenkins container to Minikube's Docker network,
2. reads Minikube's admin kubeconfig and points it at the node IP,
3. loads the new image into Minikube (or lets it pull from Docker Hub if you pushed it),
4. applies [`k8s/`](../k8s) and waits for the rolling update of 2 replicas.

Open the deployed app:

```powershell
minikube service oulad-analytics -n oulad     # opens the browser
kubectl -n oulad get pods,svc
```

## 5. Screenshots for the report

| # | Where | What it shows |
|---|---|---|
| 1 | Job page → *Stage View* (or *Pipeline Overview*) | all stages green |
| 2 | Build → *Console Output* | tests passing, image built, health check |
| 3 | Build → *Tests* (Test Result) | 42 tests, 0 failures |
| 4 | Build → *Coverage report* | coverage ≥ 80 % |
| 5 | Build → *Changes* / job *Git Polling Log* | build started by a new commit |
| 6 | Docker Desktop → *Images* (or `docker image ls oulad-analytics`) | tagged images |
| 7 | <http://localhost:8081> | deployed test server with build number and commit |
| 8 | `kubectl -n oulad get pods,svc` + `minikube service` page | Kubernetes deployment |
| 9 | Docker Hub repository page | pushed tags (if used) |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `permission denied ... /var/run/docker.sock` | Jenkins must run as root to use the socket on Docker Desktop (already set in `docker-compose.yml`). Recreate: `docker compose up -d --force-recreate`. |
| `docker: 'buildx' is not a docker command` | the Jenkins image was built before the buildx plugin was added: `docker compose build --no-cache`. |
| Port 8080 or 8081 already in use | stop the other program or change the left number in `ports:` / `TEST_PORT`. |
| Build does not start after a push | the first build must be started by hand (step 2); check *Git Polling Log*. |
| `docker exec minikube ...: No such container` | Minikube is not running or uses another driver: `minikube start --driver=docker`. |
| Pods stay in `ErrImageNeverPull`/`ImagePullBackOff` | Minikube uses containerd: recreate it with `--container-runtime=docker`, or push the image to Docker Hub. |

Stop everything: `docker compose down` (in `jenkins/`), `docker rm -f oulad-test`, `minikube stop`.
