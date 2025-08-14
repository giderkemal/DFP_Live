pipeline {
    agent { label 'windows-VS2022' }

    environment {
        GIT_COMMIT_SHORT = "${env.GIT_COMMIT.take(7)}"
        
        APP_NAME = "vapafpdf-${env.GIT_BRANCH == 'master' ? 'prd' : env.GIT_BRANCH}-streamlit-genai"
        ARTIFACTORY_URL = "https://art.pmideep.com/artifactory/"
        ARTIFACTORY_REPO = "vapafpdf-generic-${env.GIT_BRANCH == 'master' ? 'prod' : env.GIT_BRANCH}"
        ARTIFACTORY_TOKEN = "artifactory-vapafpdf-${env.GIT_BRANCH == 'master' ? 'prod' : env.GIT_BRANCH}"
        ARTIFACTORY_PACKAGE = "${APP_NAME}_${env.BUILD_NUMBER}.zip"
        ARTIFACTORY_LATEST_PACKAGE = "${APP_NAME}.zip"
        AWS_S_ACCOUNT_CRED = "Jenkins-aws-${env.GIT_BRANCH == 'master' ? 'prd' : env.GIT_BRANCH}-s-account-user"

        NODEJS_HOME = tool name: 'node'
        EC2_HOST = "172.27.198.13"
        DESTINATION_PATH = "D:/dev/repos/genai/"
    }

    tools { 
        nodejs 'node' 
    }

    stages {
        stage('Checkout') {
            steps {
                deleteDir()
                checkout scm
                echo "Checked out code from ${env.GIT_BRANCH}"
            }
        }

        stage('Package') {
            steps {
                echo "Zipping build output..."
                powershell """
                    if (Test-Path "${env.ARTIFACTORY_PACKAGE}") {
                        Remove-Item "${env.ARTIFACTORY_PACKAGE}" -Force
                    }
                    Compress-Archive -Path "${env.WORKSPACE}\\*" -DestinationPath "${env.ARTIFACTORY_PACKAGE}" -Force
                """
            }
        }

        stage('Publish to Artifactory') {
            steps {
                rtServer (
                    id: 'pipeline_arti',
                    url: "${env.ARTIFACTORY_URL}",
                    credentialsId: "${env.ARTIFACTORY_TOKEN}",
                    timeout: 300
                )
                rtUpload (
                    serverId: "pipeline_arti",
                    spec: """
                    {
                        "files": [
                            {
                                "pattern": "${env.WORKSPACE}/${env.ARTIFACTORY_PACKAGE}",
                                "target": "${env.ARTIFACTORY_REPO}/${env.ARTIFACTORY_PACKAGE}"
                            },
                            {
                                "pattern": "${env.WORKSPACE}/${env.ARTIFACTORY_PACKAGE}",
                                "target": "${env.ARTIFACTORY_REPO}/${env.ARTIFACTORY_LATEST_PACKAGE}"
                            }
                        ]
                    }
                    """
                )
            }
        }

        stage('Deploy') {
            when {
                anyOf { branch 'dev'; branch 'qa'; branch 'master' }
            }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: AWS_S_ACCOUNT_CRED,
                        usernameVariable: 'EC2_USERNAME',
                        passwordVariable: 'EC2_PASSWORD'
                    )
                ]) {
                    powershell '''
                        try {
                            # Convert password to secure string
                            $securePassword = ConvertTo-SecureString -String $env:EC2_PASSWORD -AsPlainText -Force
                            $credential = New-Object System.Management.Automation.PSCredential($env:EC2_USERNAME, $securePassword)

                            # Set up session options
                            $sessionOptions = New-PSSessionOption -SkipCACheck -SkipCNCheck -SkipRevocationCheck

                            # Establish remote session
                            Write-Host "Connecting to 172.27.198.13..."
                            $session = New-PSSession -ComputerName 172.27.198.13 -Credential $credential -UseSSL -SessionOption $sessionOptions -ErrorAction Stop

                            # Copy files
                            Write-Host "Copying files to D:/dev/repos/genai/..."
                            Copy-Item -Path "$env:WORKSPACE\\*" -Destination "D:/dev/repos/genai/" -ToSession $session -Recurse -Force

                            Write-Host "Deployment completed successfully!"
                            exit 0
                        } catch {
                            Write-Host "ERROR: $_"
                            exit 1
                        } finally {
                            if ($session) {
                                Remove-PSSession $session
                            }
                        }
                    '''
                }
            }
        }
    }
}