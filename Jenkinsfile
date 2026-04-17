pipeline {
    agent any

    stages {

        stage('Install') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install flake8 bandit'
            }
        }

        stage('Lint') {
            steps {
                sh 'flake8 .'
            }
        }

        stage('Security') {
            steps {
                sh 'bandit -r .'
            }
        }

        stage('Test') {
            steps {
                sh 'pytest tests/'
            }
        }

        stage('Build') {
            steps {
                sh 'docker build -t praneeth7975/devops-app .'
            }
        }

        stage('Push') {
            steps {
                sh 'docker push praneeth7975/devops-app'
            }
        }
    }
}
