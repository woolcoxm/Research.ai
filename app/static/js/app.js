
// AI Research System - Frontend JavaScript

class AIResearchSystem {
    constructor() {
        this.currentSession = null;
        this.currentPlan = null;
        this.displayedMessages = new Set(); // Track displayed messages to avoid duplicates
        this.init();
    }

    init() {
        this.checkSystemStatus();
        this.loadPlans();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Prompt input validation
        document.getElementById('user-prompt').addEventListener('input', (e) => {
            const startButton = document.getElementById('start-research');
            startButton.disabled = e.target.value.trim().length === 0;
        });

        // Start research
        document.getElementById('start-research').addEventListener('click', () => {
            this.startResearch();
        });

        // Next conversation round
        document.getElementById('next-round').addEventListener('click', () => {
            this.nextConversationRound();
        });

        // Generate plan
        document.getElementById('generate-plan').addEventListener('click', () => {
            this.generatePlan();
        });

        // Refresh plans
        document.getElementById('refresh-plans').addEventListener('click', () => {
            this.loadPlans();
        });

        // Export plan
        document.getElementById('export-plan').addEventListener('click', () => {
            this.exportPlan();
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            const statusElement = document.getElementById('system-status');
            
            if (data.system_initialized) {
                statusElement.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle"></i> System Ready
                        <div class="mt-2">
                            <small class="text-muted">
                                DeepSeek: ${data.components.deepseek ? '✅' : '❌'} | 
                                Ollama: ${data.components.ollama ? '✅' : '❌'} | 
                                Serper: ${data.components.serper ? '✅' : '❌'}
                            </small>
                        </div>
                    </div>
                `;
            } else {
                statusElement.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle"></i> System Partially Initialized
                        <div class="mt-2">
                            <small class="text-muted">
                                Check your configuration and ensure all services are running.
                            </small>
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to check system status:', error);
            document.getElementById('system-status').innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-times-circle"></i> System Unavailable
                </div>
            `;
        }
    }

    async startResearch() {
        const prompt = document.getElementById('user-prompt').value.trim();
        if (!prompt) return;

        const startButton = document.getElementById('start-research');
        startButton.disabled = true;
        startButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting Full Automation...';

        // Hide manual controls
        document.getElementById('next-round').style.display = 'none';
        document.getElementById('generate-plan').style.display = 'none';

        try {
            const response = await fetch('/research', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: prompt })
            });

            const data = await response.json();

            if (response.ok) {
                this.currentSession = data.session_id;
                
                // Clear previous conversation messages
                const conversationContainer = document.getElementById('conversation-messages');
                conversationContainer.innerHTML = '';
                this.displayedMessages.clear(); // Reset message tracking
                
                this.showProgressSection();
                this.updateProgress(data.research_context);
                
                // Show automation status
                this.showMessage('Full automation started! The system will now research, discuss, and generate a development plan automatically.', 'success');
                
                // Start polling for updates
                this.startPollingForUpdates();
            } else {
                throw new Error(data.error || 'Failed to start research');
            }
        } catch (error) {
            console.error('Research start failed:', error);
            this.showMessage(`Failed to start research: ${error.message}`, 'error');
            startButton.disabled = false;
            startButton.innerHTML = '<i class="fas fa-play"></i> Start Research';
        }
    }

    startPollingForUpdates() {
        // Poll for updates every 3 seconds for real-time feedback
        this.pollingInterval = setInterval(async () => {
            await this.checkForUpdates();
        }, 3000);
    }

    async checkForUpdates() {
        try {
            // First check session status for real-time progress
            if (this.currentSession) {
                const statusResponse = await fetch(`/session/${this.currentSession}/status`);
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    console.log('Session status update:', statusData);
                    
                    // Update progress with real data
                    this.updateProgress(statusData);
                    
                    // Show status updates in conversation area
                    this.showStatusUpdate(statusData);
                    
                    // Check if automation completed
                    if (statusData.completed) {
                        clearInterval(this.pollingInterval);
                        this.showMessage('Automation completed! Development plan generated and saved.', 'success');
                        this.loadPlans();
                        
                        // Show completion status
                        document.getElementById('progress-bar').style.width = '100%';
                        document.getElementById('progress-text').textContent = '100%';
                        
                        // Update conversation stages to show completion
                        this.updateConversationStages('completed', statusData.conversation_round);
                        
                        return; // Don't check for plans separately
                    }
                    
                    // Check if automation failed
                    if (statusData.failed) {
                        clearInterval(this.pollingInterval);
                        this.showMessage(`Automation failed: ${statusData.error}`, 'error');
                        return;
                    }
                }
            }

            // Then check for completed plans as fallback
            const plansResponse = await fetch('/plans');
            const plansData = await plansResponse.json();

            if (plansResponse.ok && plansData.plans.length > 0) {
                // Check if we have a new plan from the current session
                const latestPlan = plansData.plans[0];
                if (latestPlan.session_id === this.currentSession) {
                    // Automation completed!
                    clearInterval(this.pollingInterval);
                    this.showMessage('Automation completed! Development plan generated and saved.', 'success');
                    this.loadPlans();
                    
                    // Show completion status
                    document.getElementById('progress-bar').style.width = '100%';
                    document.getElementById('progress-text').textContent = '100%';
                    
                    // Update conversation stages to show completion
                    this.updateConversationStages('completed', 4);
                    
                    // Display the completed plan
                    this.viewPlan(latestPlan.filename);
                }
            }
        } catch (error) {
            console.error('Failed to check for updates:', error);
        }
    }

    showStatusUpdate(statusData) {
        const container = document.getElementById('conversation-messages');
        
        // Show latest LLM messages if available
        if (statusData.latest_messages && statusData.latest_messages.length > 0) {
            statusData.latest_messages.forEach(message => {
                this.displayMessage(message);
            });
        }
        
        // Create a status message element for progress updates
        const statusElement = document.createElement('div');
        statusElement.className = 'status-update p-2 mb-2 rounded bg-light';
        
        const stageNames = {
            'initial_research': 'Initial Research',
            'deepseek_analysis': 'DeepSeek Analysis',
            'ollama_review': 'Ollama Review',
            'discussion_round': 'Discussion Round',
            'targeted_research': 'Targeted Research',
            'plan_generation': 'Plan Generation',
            'completed': 'Completed'
        };
        
        const currentStage = stageNames[statusData.current_stage] || statusData.current_stage;
        
        statusElement.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <small class="text-muted">
                    <i class="fas fa-sync-alt fa-spin me-1"></i>
                    Round ${statusData.conversation_round}: ${currentStage}
                </small>
                <small class="text-muted">
                    Maturity: ${Math.round(statusData.context_maturity * 100)}% | Messages: ${statusData.message_count} | Searches: ${statusData.search_count}
                </small>
            </div>
        `;
        
        // Only add status if this is a new status (not duplicate)
        const lastStatus = container.lastElementChild;
        if (!lastStatus || !lastStatus.textContent.includes(`Round ${statusData.conversation_round}: ${currentStage}`)) {
            container.appendChild(statusElement);
            container.scrollTop = container.scrollHeight;
        }
    }

    displayMessage(message) {
        const container = document.getElementById('conversation-messages');
        
        // Check if this message is already displayed using timestamp
        const messageKey = `${message.llm_type}-${message.timestamp}`;
        if (this.displayedMessages.has(messageKey)) {
            return; // Message already displayed
        }
        
        this.displayedMessages.add(messageKey);
        
        // Calculate step number
        const stepNumber = this.displayedMessages.size;
        
        const messageElement = document.createElement('div');
        messageElement.className = 'llm-message';
        messageElement.dataset.timestamp = message.timestamp;
        
        const llmName = message.llm_type === 'deepseek' ? 'DeepSeek' : 'Ollama';
        const llmClass = message.llm_type === 'deepseek' ? 'deepseek' : 'ollama';
        const uniqueId = `message-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        // Create a preview of the message content (first 100 characters)
        const preview = this.createMessagePreview(message.content);
        
        messageElement.innerHTML = `
            <div class="llm-message-header ${llmClass}" onclick="window.aiResearchSystem.toggleMessage('${uniqueId}')">
                <div class="d-flex justify-content-between align-items-center">
                    <div class="d-flex align-items-center">
                        <span class="badge bg-secondary me-2" style="font-size: 0.7rem;">Step ${stepNumber}</span>
                        <span class="llm-badge ${llmClass}">${llmName}</span>
                        <small class="text-muted ms-3">${new Date(message.timestamp).toLocaleTimeString()}</small>
                    </div>
                    <i class="fas fa-chevron-down llm-toggle-icon" id="icon-${uniqueId}"></i>
                </div>
                <div class="message-preview" id="preview-${uniqueId}">${preview}</div>
            </div>
            <div class="llm-message-content" id="content-${uniqueId}">
                ${this.formatMessageContent(message.content)}
            </div>
        `;
        
        container.appendChild(messageElement);
        container.scrollTop = container.scrollHeight;
    }

    createMessagePreview(content) {
        // Remove HTML tags and create a clean preview
        const cleanContent = content.replace(/<[^>]*>/g, '').replace(/\n+/g, ' ').trim();
        const preview = cleanContent.length > 120 ? cleanContent.substring(0, 120) + '...' : cleanContent;
        return preview;
    }

    toggleMessage(messageId) {
        const content = document.getElementById(`content-${messageId}`);
        const icon = document.getElementById(`icon-${messageId}`);
        const preview = document.getElementById(`preview-${messageId}`);
        
        if (content.classList.contains('expanded')) {
            // Collapse
            content.classList.remove('expanded');
            content.style.maxHeight = '0';
            icon.classList.remove('expanded');
            preview.style.display = 'block';
        } else {
            // Expand
            content.classList.add('expanded');
            content.style.maxHeight = content.scrollHeight + 'px';
            icon.classList.add('expanded');
            preview.style.display = 'none';
        }
    }

    expandAllMessages() {
        const contents = document.querySelectorAll('.llm-message-content');
        const icons = document.querySelectorAll('.llm-toggle-icon');
        const previews = document.querySelectorAll('.message-preview');
        
        contents.forEach(content => {
            content.classList.add('expanded');
            content.style.maxHeight = content.scrollHeight + 'px';
        });
        
        icons.forEach(icon => {
            icon.classList.add('expanded');
        });
        
        previews.forEach(preview => {
            preview.style.display = 'none';
        });
    }

    collapseAllMessages() {
        const contents = document.querySelectorAll('.llm-message-content');
        const icons = document.querySelectorAll('.llm-toggle-icon');
        const previews = document.querySelectorAll('.message-preview');
        
        contents.forEach(content => {
            content.classList.remove('expanded');
            content.style.maxHeight = '0';
        });
        
        icons.forEach(icon => {
            icon.classList.remove('expanded');
        });
        
        previews.forEach(preview => {
            preview.style.display = 'block';
        });
    }

    formatMessageContent(content) {
        // Basic formatting for better readability
        return content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    async nextConversationRound() {
        const nextButton = document.getElementById('next-round');
        nextButton.disabled = true;
        nextButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

        try {
            const response = await fetch('/conversation/next', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();

            if (response.ok) {
                this.updateProgress(data.research_context);
                this.displayLatestMessages(data.latest_messages);
                
                if (data.current_stage === 'plan_generation') {
                    document.getElementById('generate-plan').disabled = false;
                    document.getElementById('next-round').disabled = true;
                }
                
                this.showMessage('Conversation round completed!', 'success');
            } else {
                throw new Error(data.error || 'Failed to execute conversation round');
            }
        } catch (error) {
            console.error('Conversation round failed:', error);
            this.showMessage(`Failed to execute conversation round: ${error.message}`, 'error');
        } finally {
            nextButton.disabled = false;
            nextButton.innerHTML = '<i class="fas fa-forward"></i> Next Round';
        }
    }

    async generatePlan() {
        const generateButton = document.getElementById('generate-plan');
        generateButton.disabled = true;
        generateButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

        try {
            const response = await fetch('/plan/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();

            if (response.ok) {
                this.currentPlan = data.development_plan;
                this.updateProgress(data.development_plan.research_metrics);
                this.displayPlan(data.development_plan);
                this.loadPlans(); // Refresh the plans list
                this.showMessage('Development plan generated successfully!', 'success');
            } else {
                throw new Error(data.error || 'Failed to generate plan');
            }
        } catch (error) {
            console.error('Plan generation failed:', error);
            this.showMessage(`Failed to generate plan: ${error.message}`, 'error');
        } finally {
            generateButton.disabled = true; // Keep disabled after generation
            generateButton.innerHTML = '<i class="fas fa-file-code"></i> Generate Plan';
        }
    }

    showProgressSection() {
        document.getElementById('progress-section').classList.remove('d-none');
        document.getElementById('automation-status').classList.remove('d-none');
        document.getElementById('manual-controls').classList.add('d-none');
        document.getElementById('conversation-section').classList.remove('d-none');
    }

    updateProgress(researchContext) {
        // Update progress bar
        const progress = Math.min(100, researchContext.context_maturity * 100);
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        
        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${Math.round(progress)}%`;

        // Update conversation stages
        this.updateConversationStages(researchContext.current_stage, researchContext.conversation_round);

        // Update quality gates
        this.updateQualityGates(researchContext.quality_gates_passed);

        // Update research metrics
        this.updateResearchMetrics(researchContext);
    }

    updateConversationStages(currentStage, currentRound) {
        const stages = [
            { id: 'initial_research', label: 'Initial Research' },
            { id: 'deepseek_analysis', label: 'DeepSeek Analysis' },
            { id: 'ollama_review', label: 'Ollama Review' },
            { id: 'discussion_round', label: 'Discussion' },
            { id: 'targeted_research', label: 'Targeted Research' },
            { id: 'plan_generation', label: 'Plan Generation' },
            { id: 'completed', label: 'Completed' }
        ];

        const container = document.getElementById('conversation-stages');
        container.innerHTML = '';

        stages.forEach(stage => {
            const stageElement = document.createElement('span');
            stageElement.className = 'conversation-stage';
            
            if (stage.id === currentStage) {
                stageElement.classList.add('stage-active');
            } else if (this.isStageCompleted(stage.id, currentStage)) {
                stageElement.classList.add('stage-completed');
            } else {
                stageElement.classList.add('stage-pending');
            }

            stageElement.textContent = stage.label;
            container.appendChild(stageElement);
        });
    }

    isStageCompleted(stageId, currentStage) {
        const stageOrder = [
            'initial_research', 'deepseek_analysis', 'ollama_review', 
            'discussion_round', 'targeted_research', 'plan_generation', 'completed'
        ];
        
        const currentIndex = stageOrder.indexOf(currentStage);
        const stageIndex = stageOrder.indexOf(stageId);
        
        return stageIndex < currentIndex;
    }

    updateQualityGates(passedGates) {
        const gates = [
            { id: 'initial_research', label: 'Initial Research' },
            { id: 'llm_consensus', label: 'LLM Consensus' },
            { id: 'implementation_feasibility', label: 'Feasibility' },
            { id: 'research_validation', label: 'Research Validation' },
            { id: 'final_plan_quality', label: 'Plan Quality' }
        ];

        const container = document.getElementById('quality-gates');
        container.innerHTML = '';

        gates.forEach(gate => {
            const gateElement = document.createElement('span');
            gateElement.className = 'quality-gate';
            
            if (passedGates.includes(gate.id)) {
                gateElement.classList.add('gate-passed');
                gateElement.innerHTML = `<i class="fas fa-check"></i> ${gate.label}`;
            } else {
                gateElement.classList.add('gate-pending');
                gateElement.innerHTML = `<i class="fas fa-clock"></i> ${gate.label}`;
            }

            container.appendChild(gateElement);
        });
    }

    updateResearchMetrics(researchContext) {
        const metricsElement = document.getElementById('research-metrics');
        
        metricsElement.innerHTML = `
            <div class="row text-center">
                <div class="col-6 mb-3">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title">Context Maturity</h6>
                            <h4 class="text-primary">${Math.round(researchContext.context_maturity * 100)}%</h4>
                        </div>
                    </div>
                </div>
                <div class="col-6 mb-3">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title">Conversation Rounds</h6>
                            <h4 class="text-success">${researchContext.conversation_round}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title">Messages</h6>
                            <h4 class="text-info">${researchContext.message_count || 0}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title">Searches</h6>
                            <h4 class="text-warning">${researchContext.search_count || 0}</h4>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    displayLatestMessages(messages) {
        const container = document.getElementById('conversation-messages');
        console.log('Displaying messages:', messages);
        
        // Clear existing messages to avoid duplicates
        container.innerHTML = '';
        
        messages.forEach(message => {
            const messageElement = document.createElement('div');
            messageElement.className = `message-${message.llm_type} p-3 mb-3 rounded`;
            
            const llmName = message.llm_type === 'deepseek' ? 'DeepSeek' : 'Ollama';
            const icon = message.llm_type === 'deepseek' ? 'fas fa-brain' : 'fas fa-code';
            const badgeColor = message.llm_type === 'deepseek' ? 'bg-primary' : 'bg-success';
            
            messageElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <span class="badge ${badgeColor}">
                        <i class="${icon} me-1"></i>${llmName}
                    </span>
                    <small class="text-muted">${new Date(message.timestamp).toLocaleTimeString()}</small>
                </div>
                <div class="message-content">${this.formatMessageContent(message.content)}</div>
            `;
            
            container.appendChild(messageElement);
        });

        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    }

    formatMessageContent(content) {
        // Simple formatting - replace line breaks with <br> tags
        return content.replace(/\n/g, '<br>');
    }

    displayPlan(plan) {
        const planModal = new bootstrap.Modal(document.getElementById('planModal'));
        const planContent = document.getElementById('plan-content');
        
        planContent.innerHTML = `
            <div class="plan-card p-4 mb-4">
                <h4 class="text-primary">${plan.project_name}</h4>
                <p class="text-muted">${plan.user_prompt}</p>
                
                <div class="mb-3">
                    <span class="badge feasibility-badge ${plan.feasibility_assessment.feasibility_score > 0.7 ? 'bg-success' : plan.feasibility_assessment.feasibility_score > 0.5 ? 'bg-warning' : 'bg-danger'}">
                        Feasibility: ${(plan.feasibility_assessment.feasibility_score * 100).toFixed(0)}%
                    </span>
                </div>
                
                <div class="development-plan-content">
                    ${this.formatMessageContent(plan.development_plan)}
                </div>
                
                <div class="mt-4">
                    <h6>Research Metrics</h6>
                    <div class="row">
                        <div class="col-md-3">
                            <small class="text-muted">Searches: ${plan.research_metrics.total_searches}</small>
                        </div>
                        <div class="col-md-3">
                            <small class="text-muted">Insights: ${plan.research_metrics.key_insights}</small>
                        </div>
                        <div class="col-md-3">
                            <small class="text-muted">Rounds: ${plan.research_metrics.conversation_rounds}</small>
                        </div>
                        <div class="col-md-3">
                            <small class="text-muted">Maturity: ${(plan.research_metrics.context_maturity * 100).toFixed(0)}%</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        planModal.show();
    }

    async loadPlans() {
        try {
            const response = await fetch('/plans');
            const data = await response.json();

            if (response.ok) {
                this.displayPlans(data.projects, data.statistics);
            } else {
                throw new Error(data.error || 'Failed to load plans');
            }
        } catch (error) {
            console.error('Failed to load plans:', error);
            document.getElementById('plans-list').innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> Failed to load plans
                </div>
            `;
        }
    }

    displayPlans(projects, statistics) {
        const container = document.getElementById('plans-list');
        
        if (projects.length === 0) {
            container.innerHTML = `
                <p class="text-muted text-center">No development plans yet</p>
                <div class="text-center">
                    <small class="text-muted">Start a research session to generate your first plan!</small>
                </div>
            `;
            return;
        }

        let html = `
            <div class="mb-3">
                <small class="text-muted">
                    Total: ${projects.length} projects | 
                    Multi-document: ${projects.filter(p => p.type === 'multi_document').length} | 
                    Recent: ${statistics.recent_plans || 0}
                </small>
            </div>
        `;

        projects.slice(0, 10).forEach(project => {
            const isMultiDoc = project.type === 'multi_document';
            const feasibilityPercent = project.feasibility_score ? (project.feasibility_score * 100).toFixed(0) : 'N/A';
            const badgeClass = project.feasibility_score > 0.7 ? 'bg-success' : project.feasibility_score > 0.5 ? 'bg-warning' : 'bg-secondary';
            
            // Build documents section
            let documentsHtml = '';
            if (isMultiDoc && project.documents) {
                documentsHtml = `
                    <div class="mt-2">
                        <small class="text-muted d-block mb-1">
                            <i class="fas fa-file-alt"></i> ${project.documents.length} Documents:
                        </small>
                        <div class="row">
                `;
                
                project.documents.forEach(doc => {
                    const sizeKB = (doc.size / 1024).toFixed(1);
                    documentsHtml += `
                        <div class="col-md-6 mb-1">
                            <small>
                                <i class="fas fa-download text-primary"></i>
                                <a href="${doc.download_url}" class="text-decoration-none" download>
                                    ${doc.title}
                                </a>
                                <span class="text-muted">(${sizeKB} KB)</span>
                            </small>
                        </div>
                    `;
                });
                
                documentsHtml += `
                        </div>
                    </div>
                `;
            } else if (project.documents && project.documents.length > 0) {
                // Legacy single document
                const doc = project.documents[0];
                documentsHtml = `
                    <div class="mt-2">
                        <small>
                            <i class="fas fa-download text-primary"></i>
                            <a href="${doc.download_url}" class="text-decoration-none" download>
                                Download ${doc.title}
                            </a>
                        </small>
                    </div>
                `;
            }
            
            html += `
                <div class="card mb-3 ${isMultiDoc ? 'border-success' : ''}">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <div class="d-flex align-items-center mb-1">
                                    <h6 class="card-title mb-0 me-2">${project.project_name}</h6>
                                    ${isMultiDoc ? '<span class="badge bg-success text-white">Multi-Doc</span>' : '<span class="badge bg-secondary text-white">Legacy</span>'}
                                </div>
                                <small class="text-muted d-block mb-2">
                                    <i class="fas fa-calendar"></i> ${new Date(project.generated_at).toLocaleDateString()} | 
                                    <i class="fas fa-quote-left"></i> ${project.user_prompt.substring(0, 60)}${project.user_prompt.length > 60 ? '...' : ''}
                                </small>
                                ${documentsHtml}
                            </div>
                            <div class="text-end ms-3">
                                ${project.feasibility_score ? `<span class="badge ${badgeClass} mb-1">${feasibilityPercent}%</span><br>` : ''}
                                <small>
                                    <a href="#" class="view-project text-primary" data-project='${JSON.stringify(project)}'>
                                        <i class="fas fa-eye"></i> View Details
                                    </a>
                                </small>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;

        // Add event listeners for view project links
        container.querySelectorAll('.view-project').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const projectData = JSON.parse(e.target.dataset.project);
                this.viewProject(projectData);
            });
        });
    }

    viewProject(projectData) {
        // Display project details in a modal or dedicated view
        const modalContent = document.querySelector('#plan-modal .modal-body');
        
        let content = `
            <h5>${projectData.project_name}</h5>
            <p><strong>Request:</strong> ${projectData.user_prompt}</p>
            <p><strong>Generated:</strong> ${new Date(projectData.generated_at).toLocaleDateString()}</p>
            <p><strong>Type:</strong> ${projectData.type === 'multi_document' ? 'Multi-Document Project' : 'Single Document'}</p>
        `;
        
        if (projectData.documents && projectData.documents.length > 0) {
            content += `<h6>Available Documents:</h6><div class="list-group">`;
            
            projectData.documents.forEach(doc => {
                const sizeKB = doc.size ? (doc.size / 1024).toFixed(1) : 'Unknown';
                content += `
                    <div class="list-group-item d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${doc.title}</strong>
                            <br><small class="text-muted">${doc.filename} (${sizeKB} KB)</small>
                        </div>
                        <a href="${doc.download_url}" class="btn btn-outline-primary btn-sm" download>
                            <i class="fas fa-download"></i> Download
                        </a>
                    </div>
                `;
            });
            
            content += `</div>`;
        }
        
        modalContent.innerHTML = content;
        
        // Show the modal
        const modal = new bootstrap.Modal(document.getElementById('plan-modal'));
        modal.show();
    }

    async viewPlan(filename) {
        try {
            const response = await fetch(`/plans/${filename}`);
            const plan = await response.json();

            if (response.ok) {
                this.currentPlan = plan;
                this.displayPlan(plan);
            } else {
                throw new Error(plan.error || 'Failed to load plan');
            }
        } catch (error) {
            console.error('Failed to view plan:', error);
            this.showMessage(`Failed to load plan: ${error.message}`, 'error');
        }
    }

    async exportPlan() {
        if (!this.currentPlan) return;

        try {
            const filename = this.currentPlan.file_metadata?.filename;
            if (!filename) {
                throw new Error('No plan filename available');
            }

            window.location.href = `/plans/${filename}/export`;
        } catch (error) {
            console.error('Failed to export plan:', error);
            this.showMessage(`Failed to export plan: ${error.message}`, 'error');
        }
    }

    showMessage(message, type = 'info') {
        // Create a simple toast notification
        const alertClass = type === 'error' ? 'alert-danger' :
                          type === 'success' ? 'alert-success' : 'alert-info';
        
        const alertElement = document.createElement('div');
        alertElement.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
        alertElement.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
        
        alertElement.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertElement);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertElement.parentNode) {
                alertElement.remove();
            }
        }, 5000);
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiResearchSystem = new AIResearchSystem();
});
