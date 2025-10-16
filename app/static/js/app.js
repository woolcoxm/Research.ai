// AI Research System - Streamlined 5-Stage Workflow

class AIResearchSystem {
    constructor() {
        this.currentSession = null;
        this.pollInterval = null;
        this.activityCount = 0;
        this.init();
    }

    init() {
        this.setupEventListeners();
        console.log('🚀 AI Research System initialized - 5-Stage Streamlined Workflow');
    }

    setupEventListeners() {
        // Prompt input
        const promptInput = document.getElementById('user-prompt');
        const startBtn = document.getElementById('start-research');
        
        promptInput.addEventListener('input', (e) => {
            startBtn.disabled = e.target.value.trim().length === 0;
        });

        // Start research
        startBtn.addEventListener('click', () => {
            this.startResearch();
        });
    }

    async startResearch() {
        const prompt = document.getElementById('user-prompt').value.trim();
        if (!prompt) return;

        // Disable start button
        document.getElementById('start-research').disabled = true;

        // Show progress sections
        document.getElementById('progress-overview').classList.remove('hidden');
        document.getElementById('stages-section').classList.remove('hidden');
        document.getElementById('activity-section').classList.remove('hidden');

        try {
            // Start research session
            const response = await fetch('/api/start_research', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt })
            });

            const data = await response.json();
            
            if (data.success) {
                this.currentSession = data.session_id;
                console.log('🎯 SESSION STARTED:', data.session_id);
                console.log('📋 To monitor: python monitor_session.py', data.session_id);
                this.addActivity('system', 'Research Started', 'Initializing 5-stage workflow...');
                
                // Start polling for updates
                this.startPolling();
            } else {
                alert('Failed to start research: ' + data.error);
                document.getElementById('start-research').disabled = false;
            }
        } catch (error) {
            console.error('Error starting research:', error);
            alert('Error starting research. Check console for details.');
            document.getElementById('start-research').disabled = false;
        }
    }

    startPolling() {
        // Poll every 2 seconds
        this.pollInterval = setInterval(() => {
            this.pollStatus();
        }, 2000);
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    async pollStatus() {
        if (!this.currentSession) return;

        try {
            const response = await fetch(`/api/status/${this.currentSession}`);
            const data = await response.json();

            console.log('📊 Poll Response:', {
                stage: data.stage,
                round: data.conversation_round,
                docs_planned: data.documents_planned,
                docs_completed: data.documents_completed,
                docs_total: data.documents_total
            });

            if (data.success) {
                this.updateUI(data);

                // Check if completed
                if (data.stage === 'completed') {
                    console.log('✅ Research marked as COMPLETED');
                    this.stopPolling();
                    this.onResearchComplete(data);
                } else {
                    console.log(`⟳ Stage ${data.stage} in progress...`);
                }
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }

    updateUI(data) {
        // Update metrics
        this.updateMetrics(data);
        
        // Update stages
        this.updateStages(data);
        
        // Add activity if there's a status update
        if (data.current_activity) {
            const llm = data.current_activity.llm || 'system';
            const activity = data.current_activity.activity || '';
            const details = data.current_activity.details || '';
            this.addActivity(llm.toLowerCase(), activity, details);
        }
    }

    updateMetrics(data) {
        // Research Queries
        document.getElementById('metric-queries').textContent = data.research_queries || 0;
        const queriesStatus = document.getElementById('status-queries');
        if (data.research_queries > 0) {
            queriesStatus.textContent = '✓ Complete';
            queriesStatus.className = 'metric-status complete';
        }

        // Sources
        document.getElementById('metric-sources').textContent = data.sources_collected || 0;
        const sourcesStatus = document.getElementById('status-sources');
        if (data.sources_collected > 0) {
            sourcesStatus.textContent = '✓ Complete';
            sourcesStatus.className = 'metric-status complete';
        }

        // Documents Planned
        document.getElementById('metric-docs-planned').textContent = data.documents_planned || 0;
        const docsPlannedStatus = document.getElementById('status-docs-planned');
        if (data.documents_planned > 0) {
            docsPlannedStatus.textContent = '✓ Complete';
            docsPlannedStatus.className = 'metric-status complete';
        }

        // Documents Written
        const docsWritten = data.documents_completed || 0;
        const docsTotal = data.documents_total || 0;
        document.getElementById('metric-docs-written').textContent = `${docsWritten}/${docsTotal}`;
        const docsWrittenStatus = document.getElementById('status-docs-written');
        if (docsWritten > 0) {
            if (docsWritten === docsTotal) {
                docsWrittenStatus.textContent = '✓ All Complete';
                docsWrittenStatus.className = 'metric-status complete';
            } else {
                docsWrittenStatus.textContent = `⟳ Writing ${docsWritten}/${docsTotal}`;
                docsWrittenStatus.className = 'metric-status active';
            }
        }
    }

    updateStages(data) {
        const currentStage = data.stage;
        const stageOrder = [
            'research_planning',
            'web_research',
            'research_analysis',
            'document_planning',
            'document_writing'
        ];

        const currentIndex = stageOrder.indexOf(currentStage);

        stageOrder.forEach((stageName, index) => {
            const stageElement = document.querySelector(`.stage-item[data-stage="${stageName}"]`);
            if (!stageElement) return;

            // Remove all status classes
            stageElement.classList.remove('pending', 'active', 'completed');

            if (index < currentIndex) {
                // Completed stages
                stageElement.classList.add('completed');
                const icon = stageElement.querySelector('.stage-icon i');
                if (icon && !icon.classList.contains('fa-check')) {
                    icon.className = 'fas fa-check';
                }
            } else if (index === currentIndex) {
                // Current stage
                stageElement.classList.add('active');
                
                // Update details
                const details = stageElement.querySelector('.stage-details');
                if (details && data.current_activity) {
                    details.textContent = data.current_activity.details || 'Processing...';
                }

                // Update progress bar
                const progressBar = stageElement.querySelector('.stage-progress-fill');
                if (progressBar) {
                    const progress = this.calculateStageProgress(data);
                    progressBar.style.width = `${progress}%`;
                }
            } else {
                // Pending stages
                stageElement.classList.add('pending');
            }
        });
    }

    calculateStageProgress(data) {
        const stage = data.stage;
        
        if (stage === 'research_planning') {
            return data.research_queries ? 100 : 50;
        } else if (stage === 'web_research') {
            const total = data.research_queries || 1;
            const collected = data.sources_collected || 0;
            return Math.min(100, (collected / (total * 5)) * 100);
        } else if (stage === 'research_analysis') {
            return 75; // Analysis is in progress
        } else if (stage === 'document_planning') {
            const planned = data.documents_planned || 0;
            return (planned / 4) * 100;
        } else if (stage === 'document_writing') {
            const completed = data.documents_completed || 0;
            const total = data.documents_total || 1;
            return (completed / total) * 100;
        }
        
        return 0;
    }

    addActivity(llm, activity, details) {
        const activityList = document.getElementById('activity-list');
        
        // Create activity item
        const item = document.createElement('div');
        item.className = `activity-item ${llm}`;
        
        const icon = this.getActivityIcon(llm);
        const time = new Date().toLocaleTimeString();
        
        item.innerHTML = `
            <div class="activity-icon">
                <i class="${icon}"></i>
            </div>
            <div class="activity-content">
                <div class="activity-title">${activity}</div>
                <div class="activity-description">${details}</div>
                <div class="activity-time">${time}</div>
            </div>
        `;
        
        // Add to top of list
        activityList.insertBefore(item, activityList.firstChild);
        
        // Update count
        this.activityCount++;
        document.getElementById('activity-count').textContent = `${this.activityCount} events`;
        
        // Limit to 50 items
        while (activityList.children.length > 50) {
            activityList.removeChild(activityList.lastChild);
        }
    }

    getActivityIcon(llm) {
        const icons = {
            'deepseek': 'fas fa-brain',
            'ollama': 'fas fa-pen-fancy',
            'serper': 'fas fa-search',
            'system': 'fas fa-cog'
        };
        return icons[llm] || 'fas fa-info-circle';
    }

    async onResearchComplete(data) {
        this.addActivity('system', 'Research Complete', 'All documents generated successfully!');
        
        // Show documents section
        document.getElementById('documents-section').classList.remove('hidden');
        
        // Load documents
        await this.loadDocuments();
    }

    async loadDocuments() {
        if (!this.currentSession) return;

        try {
            const response = await fetch(`/api/documents/${this.currentSession}`);
            const data = await response.json();

            if (data.success && data.documents) {
                this.displayDocuments(data.documents);
            }
        } catch (error) {
            console.error('Error loading documents:', error);
        }
    }

    displayDocuments(documents) {
        const documentsList = document.getElementById('documents-list');
        documentsList.innerHTML = '';

        documents.forEach((doc, index) => {
            const card = document.createElement('div');
            card.className = 'document-card';
            
            const wordCount = doc.word_count || 0;
            const charCount = doc.char_count || 0;
            
            card.innerHTML = `
                <div class="document-title">
                    <i class="fas fa-file-alt"></i> ${doc.title}
                </div>
                <div class="document-stats">
                    <div class="document-stat">
                        <i class="fas fa-file-word"></i>
                        <span>${wordCount.toLocaleString()} words</span>
                    </div>
                    <div class="document-stat">
                        <i class="fas fa-text-height"></i>
                        <span>${charCount.toLocaleString()} characters</span>
                    </div>
                </div>
                <button class="download-btn" onclick="app.downloadDocument('${doc.filename}')">
                    <i class="fas fa-download"></i> Download ${doc.filename}
                </button>
            `;
            
            documentsList.appendChild(card);
        });
    }

    async downloadDocument(filename) {
        if (!this.currentSession) return;

        try {
            const response = await fetch(`/api/download/${this.currentSession}/${filename}`);
            const blob = await response.blob();
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.addActivity('system', 'Download', `Downloaded ${filename}`);
        } catch (error) {
            console.error('Error downloading document:', error);
            alert('Error downloading document');
        }
    }
}

// Initialize app
const app = new AIResearchSystem();
