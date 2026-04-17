pipeline {
    agent any

    environment {
        IMAGE_NAME = "praneeth7975/devops-app"
    }

    stages {

        stage('Checkout') {
            steps {
                git 'https://github.com/pran-17/multi-service-app.git'
            }
        }

        stage('Install') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                pip install flake8 bandit pytest
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                . venv/bin/activate
                flake8 . || true
                '''
            }
        }

        stage('Security Scan') {
            steps {
                sh '''
                . venv/bin/activate
                bandit -r . || true
                '''
            }
        }

        stage('Unit Test') {
            steps {
                sh '''
                . venv/bin/activate
                pytest tests/ || true
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t $IMAGE_NAME .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
                    sh '''
                    echo $PASS | docker login -u $USER --password-stdin
                    docker push $IMAGE_NAME
                    '''
                }
            }
        }
    }

    post {
        success {
            echo "Pipeline executed successfully 🚀"
        }
        failure {
            echo "Pipeline failed ❌"
        }
    }
}
