// CI/CD pipeline for oulad-analytics (Jenkins declarative pipeline).
//
//   Prepare -> Build test image -> Lint -> Unit tests -> Build Docker image -> Smoke test
//           -> Push to Docker Hub (optional) -> Deploy to test server -> Deploy to Kubernetes (optional)
//
// Requirements on the Jenkins agent: git, the Docker CLI with access to a Docker daemon and,
// for the Kubernetes stage, kubectl. The image in jenkins/ provides all of them.
// Python is NOT needed on the agent: lint and tests run inside the test image, so CI uses
// exactly the same pinned environment as the image that is shipped.

pipeline {
    agent any

    triggers {
        // Check GitHub for new commits every ~2 minutes, so every pushed commit is built.
        // A Jenkins on a laptop has no public URL, so GitHub cannot call it with a webhook.
        pollSCM('H/2 * * * *')
    }

    parameters {
        string(name: 'IMAGE_NAME', defaultValue: 'oulad-analytics',
               description: 'Image repository. To push, use <dockerhub-user>/oulad-analytics.')
        booleanParam(name: 'PUSH_TO_DOCKERHUB', defaultValue: false,
                     description: 'Push the image to Docker Hub (needs the credential "dockerhub-creds").')
        booleanParam(name: 'DEPLOY_TEST_SERVER', defaultValue: true,
                     description: 'Run the new image as a test server on http://localhost:8081.')
        booleanParam(name: 'DEPLOY_TO_K8S', defaultValue: false,
                     description: 'Deploy to the local Minikube cluster (namespace "oulad").')
    }

    options {
        timestamps()
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    environment {
        TEST_IMAGE = 'oulad-analytics-test'
        TEST_PORT = '8081'
    }

    stages {
        stage('Prepare') {
            steps {
                script {
                    env.GIT_SHORT = env.GIT_COMMIT ? env.GIT_COMMIT.substring(0, 7) : 'local'
                    env.IMAGE_TAG = "${params.IMAGE_NAME}:${env.BUILD_NUMBER}-${env.GIT_SHORT}"
                    env.IMAGE_LATEST = "${params.IMAGE_NAME}:latest"
                    currentBuild.description = env.IMAGE_TAG
                }
                sh '''
                    git log -1 --pretty='Commit %h by %an: %s'
                    docker version --format 'Docker client {{.Client.Version}}, server {{.Server.Version}}'
                    echo "Image to build: $IMAGE_TAG"
                '''
            }
        }

        stage('Build test image') {
            steps {
                sh 'docker build --target test -t "$TEST_IMAGE:$BUILD_NUMBER" .'
            }
        }

        stage('Lint') {
            steps {
                sh 'docker run --rm "$TEST_IMAGE:$BUILD_NUMBER" sh -c "ruff check . && ruff format --check ."'
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    rm -rf reports && mkdir -p reports
                    set +e
                    docker run --name "oulad-tests-$BUILD_NUMBER" "$TEST_IMAGE:$BUILD_NUMBER"
                    rc=$?
                    set -e
                    docker cp "oulad-tests-$BUILD_NUMBER:/app/reports/." reports/
                    docker rm "oulad-tests-$BUILD_NUMBER" > /dev/null
                    exit $rc
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/junit.xml'
                    publishHTML(target: [reportName: 'Coverage report', reportDir: 'reports/htmlcov',
                                         reportFiles: 'index.html', keepAll: true,
                                         alwaysLinkToLastBuild: true, allowMissing: true])
                }
            }
        }

        stage('Build Docker image') {
            steps {
                sh '''
                    docker build \
                        --build-arg VERSION="$BUILD_NUMBER" \
                        --build-arg VCS_REF="$GIT_SHORT" \
                        -t "$IMAGE_TAG" -t "$IMAGE_LATEST" .
                    docker image ls "$IMAGE_NAME"
                '''
            }
        }

        stage('Smoke test image') {
            steps {
                sh '''
                    docker run --rm "$IMAGE_TAG" --version
                    rm -rf demo-results && mkdir -p demo-results
                    docker run --name "oulad-smoke-$BUILD_NUMBER" "$IMAGE_TAG" demo --out-dir /results
                    docker cp "oulad-smoke-$BUILD_NUMBER:/results/." demo-results/
                    docker rm "oulad-smoke-$BUILD_NUMBER" > /dev/null
                    test -s demo-results/summary.md
                    test -s demo-results/model_comparison.png
                '''
                archiveArtifacts artifacts: 'demo-results/**', fingerprint: true
            }
        }

        stage('Push to Docker Hub') {
            when { expression { return params.PUSH_TO_DOCKERHUB } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds',
                                                  usernameVariable: 'DOCKERHUB_USER',
                                                  passwordVariable: 'DOCKERHUB_TOKEN')]) {
                    sh '''
                        echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin
                        docker push "$IMAGE_TAG"
                        docker push "$IMAGE_LATEST"
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to test server') {
            when { expression { return params.DEPLOY_TEST_SERVER } }
            steps {
                sh '''
                    docker rm -f oulad-test > /dev/null 2>&1 || true
                    docker run -d --name oulad-test --restart unless-stopped \
                        -p "$TEST_PORT:8000" "$IMAGE_TAG" serve
                    for i in $(seq 1 30); do
                        if docker exec oulad-test python -c \
                            "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" \
                            2> /dev/null; then
                            echo "Test server is up: http://localhost:$TEST_PORT"
                            exit 0
                        fi
                        sleep 2
                    done
                    docker logs oulad-test
                    exit 1
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            when { expression { return params.DEPLOY_TO_K8S } }
            steps {
                sh '''
                    # Minikube (docker driver) runs as the container "minikube" on the same Docker host.
                    # Join its network, read its admin kubeconfig and point it at the node IP.
                    docker network connect minikube "$(cat /etc/hostname)" 2> /dev/null || true
                    MINIKUBE_IP=$(docker inspect -f '{{(index .NetworkSettings.Networks "minikube").IPAddress}}' minikube)
                    export KUBECONFIG="$WORKSPACE/.kubeconfig"
                    trap 'rm -f "$KUBECONFIG"' EXIT
                    docker exec minikube cat /etc/kubernetes/admin.conf \
                        | sed "s|server: https://.*:8443|server: https://$MINIKUBE_IP:8443|" > "$KUBECONFIG"

                    if [ "$PUSH_TO_DOCKERHUB" != "true" ]; then
                        echo "Loading $IMAGE_TAG into Minikube (no registry needed)"
                        docker save "$IMAGE_TAG" | docker exec -i minikube docker load
                    fi

                    kubectl apply -f k8s/namespace.yaml
                    sed "s|__IMAGE__|$IMAGE_TAG|g" k8s/deployment.yaml | kubectl apply -f -
                    kubectl apply -f k8s/service.yaml
                    kubectl -n oulad rollout status deployment/oulad-analytics --timeout=300s
                    kubectl -n oulad get deployment,pods,service -o wide
                    echo "Open it with: minikube service oulad-analytics -n oulad"
                '''
            }
        }
    }

    post {
        success {
            echo "Pipeline succeeded: ${env.IMAGE_TAG}"
        }
        failure {
            echo 'Pipeline failed - open the stage that is red and read its log.'
        }
        always {
            sh '''
                docker rmi "$TEST_IMAGE:$BUILD_NUMBER" > /dev/null 2>&1 || true
                docker image prune -f > /dev/null 2>&1 || true
            '''
        }
    }
}
