/**
 * LawUP Web Application Logic
 * Interactive Contract Review Dashboard & Risk Pipeline
 */

// Global State
const state = {
  currentTab: 'document', // 'document', 'clause', 'pipeline'
  documentResults: null,
  clauseResult: null,
  backendConnected: false,
  activeFilter: 'all', // 'all', 'high', 'failed', 'clean'
  searchQuery: '',
};

// Preset Sample Clauses for Fast Testing
const SAMPLE_CLAUSES = {
  indemnity: {
    title: "Uncapped Indemnification Clause",
    text: "Vendor shall defend, indemnify, and hold harmless Customer, its parent, affiliates, directors, officers, employees, and agents from and against any and all claims, liabilities, losses, damages, expenses, fines, penalties, settlements, and legal costs (including reasonable attorneys' fees) arising out of or resulting from any breach of warranty, negligence, strict liability, or willful misconduct, without any monetary cap or temporal limitation whatsoever.",
  },
  non_compete: {
    title: "Broad 2-Year Worldwide Non-Compete",
    text: "During the term of this Agreement and for a period of twenty-four (24) consecutive months following the termination or expiration thereof for any reason, Contractor shall not directly or indirectly engage in, perform services for, consult with, invest in, or otherwise assist any business, individual, or enterprise that competes with any product or service offered by Company anywhere worldwide.",
  },
  termination: {
    title: "Unilateral Immediate Termination for Convenience",
    text: "Company reserves the sole and absolute right to terminate this Agreement in whole or in part at any time, with or without cause, effective immediately upon delivery of written notice to Service Provider, without incurring any liability, penalty, or obligation to pay for unperformed milestones or non-cancelable vendor commitments.",
  },
  liability: {
    title: "Total Limitation of Liability Exclusion",
    text: "In no event shall Service Provider, its licensors, or subcontractors be liable for any indirect, incidental, punitive, special, or consequential damages, or for loss of profits, revenue, data, or business interruption, under any legal theory, and Service Provider's aggregate cumulative liability under this Agreement shall not exceed one hundred United States dollars ($100.00).",
  },
  adversarial_omission: {
    title: "Adversarial Omission Test (Dropped Escrow Obligation)",
    text: "Licensor shall deposit all source code, technical documentation, and compilation instructions for the Software into a third-party escrow account within ten (10) business days of execution, and shall verify the release condition annually at Licensor's sole expense.",
  },
  adversarial_hallucination: {
    title: "Adversarial Hallucination Test (Injected Regulatory Penalty)",
    text: "The parties agree that all disputes arising under this agreement shall be settled through binding confidential arbitration administered by the American Arbitration Association in Dover, Delaware.",
  },
};

// Preset Sample Contracts for Document Upload Simulation
const SAMPLE_CONTRACTS = {
  msa: `MASTER SERVICES AGREEMENT

ARTICLE 1. SERVICES AND WORK ORDERS
1.1 Provider agrees to provide professional technology consulting, engineering, and maintenance services as specified in Statements of Work executed under this Agreement. Provider shall dedicate competent personnel and perform all obligations in accordance with prevailing industry standards.

ARTICLE 2. COMPENSATION AND PAYMENT
2.1 Client shall pay invoices submitted by Provider within thirty (30) days of receipt. Overdue amounts shall bear interest at 1.5% per month or the maximum legal rate, whichever is lower. Client shall not withhold payment of undisputed amounts.

ARTICLE 3. TERM AND TERMINATION
3.1 This Agreement takes effect on the Effective Date and continues for twelve (12) months. Either party may terminate this Agreement without cause upon sixty (60) days prior written notice. Either party may terminate immediately if the other party commits a material breach and fails to cure such breach within thirty (30) days of notice.

ARTICLE 4. INDEMNIFICATION
4.1 Provider shall defend, indemnify, and hold harmless Client, its affiliates, officers, and employees against any third-party claims, liabilities, losses, and damages arising out of Provider's gross negligence, willful misconduct, or infringement of any registered United States patent, copyright, or trademark.

ARTICLE 5. LIMITATION OF LIABILITY
5.1 EXCEPT FOR BREACHES OF CONFIDENTIALITY OR INDEMNIFICATION OBLIGATIONS UNDER ARTICLE 4, NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES. EACH PARTY'S TOTAL AGGREGATE LIABILITY SHALL BE CAPPED AT THE TOTAL FEES PAID OR PAYABLE BY CLIENT UNDER THE APPLICABLE WORK ORDER IN THE TWELVE MONTHS PRECEDING THE CLAIM.

ARTICLE 6. NON-SOLICITATION AND NON-COMPETE
6.1 During the term of this Agreement and for twelve (12) months thereafter, neither party shall solicit for employment any employee or contractor of the other party who was directly involved in the provision or receipt of Services without written consent.`,

  nda: `MUTUAL CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT

1. DEFINITION OF CONFIDENTIAL INFORMATION
Confidential Information refers to all proprietary, non-public technical, financial, commercial, or operational information disclosed by one party ("Disclosing Party") to the other party ("Receiving Party"), whether orally, visually, or in tangible documentation marked as confidential.

2. OBLIGATIONS OF RECEIVING PARTY
Receiving Party agrees to hold Disclosing Party's Confidential Information in strict confidence and apply at least the degree of care it exercises regarding its own confidential information, but in no event less than reasonable care. Receiving Party shall not disclose Confidential Information to any third party without prior written consent.

3. EXCLUSIONS FROM CONFIDENTIALITY
The obligations of confidentiality shall not apply to information that: (a) is or becomes publicly available without breach of this Agreement; (b) was already known to Receiving Party without restriction prior to disclosure; (c) is rightfully received from a third party without duty of confidentiality; or (d) is independently developed without reference to Confidential Information.

4. INJUNCTIVE RELIEF AND REMEDIES
The parties acknowledge that unauthorized disclosure or use of Confidential Information may cause irreparable harm for which monetary damages alone would be inadequate. Disclosing Party shall be entitled to seek equitable relief, including preliminary injunction and specific performance, without proving monetary damages or posting bond.`,
};

// UI Initialization
document.addEventListener('DOMContentLoaded', () => {
  initializeTabs();
  initializeHealthCheck();
  initializeSingleClauseInspector();
  initializeDocumentAnalysis();
});

// Health Probe
async function initializeHealthCheck() {
  const badge = document.getElementById('connection-status-badge');
  const details = document.getElementById('connection-status-text');

  async function check() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        state.backendConnected = true;
        if (badge && details) {
          badge.className = 'w-2.5 h-2.5 rounded-full bg-emerald-500 status-pulse';
          details.textContent = 'API & Pipeline Online';
          details.className = 'text-xs font-medium text-emerald-700';
        }
      } else {
        throw new Error('Health returned ' + res.status);
      }
    } catch (err) {
      state.backendConnected = false;
      if (badge && details) {
        badge.className = 'w-2.5 h-2.5 rounded-full bg-amber-500';
        details.textContent = 'Backend Standby (Local Mode)';
        details.className = 'text-xs font-medium text-amber-700';
      }
    }
  }

  await check();
  setInterval(check, 30000);
}

// Tab Switching
function initializeTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-panel');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      state.currentTab = targetTab;

      tabButtons.forEach(b => b.classList.remove('active', 'border-blue-600', 'text-blue-700'));
      btn.classList.add('active', 'border-blue-600', 'text-blue-700');

      tabContents.forEach(panel => {
        if (panel.id === `tab-${targetTab}`) {
          panel.classList.remove('hidden');
        } else {
          panel.classList.add('hidden');
        }
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Single Clause Inspector
// ---------------------------------------------------------------------------
function initializeSingleClauseInspector() {
  const clauseInput = document.getElementById('single-clause-input');
  const analyzeBtn = document.getElementById('btn-analyze-clause');
  const charCounter = document.getElementById('clause-char-counter');
  const resultsContainer = document.getElementById('single-clause-results');
  const loadingIndicator = document.getElementById('single-clause-loading');
  const sampleSelector = document.getElementById('clause-sample-select');

  if (clauseInput && charCounter) {
    clauseInput.addEventListener('input', () => {
      const len = clauseInput.value.length;
      const estTokens = Math.ceil(len / 4);
      charCounter.textContent = `${len} chars (~${estTokens} tokens)`;
      if (estTokens > 1024) {
        charCounter.classList.add('text-rose-600', 'font-semibold');
      } else {
        charCounter.classList.remove('text-rose-600', 'font-semibold');
      }
    });
  }

  // Load sample clause
  if (sampleSelector && clauseInput) {
    sampleSelector.addEventListener('change', (e) => {
      const key = e.target.value;
      if (key && SAMPLE_CLAUSES[key]) {
        clauseInput.value = SAMPLE_CLAUSES[key].text;
        clauseInput.dispatchEvent(new Event('input'));
      }
    });
  }

  // Click Analyze
  if (analyzeBtn && clauseInput) {
    analyzeBtn.addEventListener('click', async () => {
      const text = clauseInput.value.trim();
      if (!text) {
        alert('Please paste or enter a legal clause to analyze.');
        return;
      }

      if (loadingIndicator) loadingIndicator.classList.remove('hidden');
      if (resultsContainer) resultsContainer.classList.add('hidden');
      analyzeBtn.disabled = true;

      try {
        const response = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ clause: text }),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();
        state.clauseResult = data;
        renderSingleClauseResult(data);
      } catch (err) {
        alert(`Analysis Error: ${err.message}\nMake sure LawUP backend and LM Studio are running.`);
      } finally {
        if (loadingIndicator) loadingIndicator.classList.add('hidden');
        if (resultsContainer) resultsContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
      }
    });
  }
}

function renderSingleClauseResult(result) {
  const container = document.getElementById('single-clause-results');
  if (!container) return;

  const originalEl = document.getElementById('sc-original-clause');
  const rewriteEl = document.getElementById('sc-simplified-rewrite');
  const risksEl = document.getElementById('sc-risks-list');
  const groundingEl = document.getElementById('sc-grounding-list');
  const verifierEl = document.getElementById('sc-verifier-summary');
  const latencyEl = document.getElementById('sc-latency-badges');

  if (originalEl) originalEl.textContent = result.original_clause;
  if (rewriteEl) rewriteEl.textContent = result.simplified_rewrite || 'No simplification produced.';

  // Risks
  if (risksEl) {
    risksEl.innerHTML = '';
    const risks = result.risks_identified || [];
    if (risks.length === 0) {
      risksEl.innerHTML = '<span class="px-2.5 py-1 text-xs rounded-full bg-emerald-100 text-emerald-800 font-medium">Standard Clause (No Critical CUAD Risk Detected)</span>';
    } else {
      risks.forEach(r => {
        const badge = document.createElement('span');
        badge.className = 'px-2.5 py-1 text-xs rounded-full bg-rose-100 text-rose-800 font-semibold border border-rose-200';
        badge.textContent = r;
        risksEl.appendChild(badge);
      });
    }
  }

  // Grounding Standards
  if (groundingEl) {
    groundingEl.innerHTML = '';
    const refs = result.grounding_references || [];
    if (refs.length === 0) {
      groundingEl.innerHTML = '<p class="text-xs text-slate-500 italic">No statutory citations retrieved.</p>';
    } else {
      refs.forEach(ref => {
        const simPct = Math.round((ref.similarity_score || 0) * 100);
        const card = document.createElement('div');
        card.className = 'p-3 bg-white rounded-lg border border-slate-200 shadow-sm text-xs space-y-1.5';
        card.innerHTML = `
          <div class="flex items-center justify-between">
            <span class="font-semibold text-slate-900">${escapeHtml(ref.title)}</span>
            <span class="px-2 py-0.5 rounded bg-blue-50 text-blue-700 font-mono font-medium">${simPct}% match</span>
          </div>
          <div class="text-slate-600 font-medium">Authority: <span class="font-normal">${escapeHtml(ref.authority || 'Legal Corpus')}</span></div>
          <p class="text-slate-500 leading-relaxed">${escapeHtml(ref.excerpt || ref.scope || '')}</p>
        `;
        groundingEl.appendChild(card);
      });
    }
  }

  // Meaning Verifier Report
  if (verifierEl) {
    const vr = result.verifier_report || {};
    const pass = vr.passes_verification;
    const badgeColor = pass ? 'bg-emerald-100 text-emerald-800 border-emerald-300' : 'bg-rose-100 text-rose-800 border-rose-300';
    const statusLabel = pass ? 'PASS: Meaning Preserved' : 'DISCREPANCY DETECTED';

    const fwdScore = vr.forward_score !== undefined ? `${Math.round(vr.forward_score * 100)}%` : 'N/A';
    const bwdScore = vr.backward_score !== undefined ? `${Math.round(vr.backward_score * 100)}%` : 'N/A';

    verifierEl.innerHTML = `
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs font-semibold uppercase text-slate-500 tracking-wider">Bidirectional NLI Verification</span>
        <span class="px-2.5 py-1 text-xs rounded-full font-bold border ${badgeColor}">${statusLabel}</span>
      </div>
      <div class="grid grid-cols-2 gap-3 text-xs mb-2">
        <div class="p-2.5 bg-slate-50 rounded border border-slate-200">
          <div class="text-slate-500 font-medium">Direction A (Original &rarr; Rewrite)</div>
          <div class="text-base font-bold text-slate-800 mt-1">${fwdScore}</div>
          <div class="text-[11px] text-slate-400">Preserves original duties</div>
        </div>
        <div class="p-2.5 bg-slate-50 rounded border border-slate-200">
          <div class="text-slate-500 font-medium">Direction B (Rewrite &rarr; Original)</div>
          <div class="text-base font-bold text-slate-800 mt-1">${bwdScore}</div>
          <div class="text-[11px] text-slate-400">No fabricated obligations</div>
        </div>
      </div>
      ${vr.failure_category ? `<div class="p-2.5 bg-rose-50 rounded border border-rose-200 text-xs text-rose-900 font-medium">Flag: ${escapeHtml(vr.failure_category)}</div>` : ''}
      ${vr.explanation ? `<p class="text-xs text-slate-600 mt-2 leading-relaxed"><strong>Judge Analysis:</strong> ${escapeHtml(vr.explanation)}</p>` : ''}
    `;
  }

  // Citation Verifier Report
  const citationEl = document.getElementById('sc-citation-summary');
  if (citationEl) {
    const cr = result.citation_report || {};
    const totalRefs = cr.total_references || (result.grounding_references ? result.grounding_references.length : 0);
    const validRefs = cr.validated_count || totalRefs;
    const droppedRefs = cr.dropped_count || 0;

    citationEl.innerHTML = `
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs font-semibold uppercase text-slate-500 tracking-wider">Citation Grounding Verifier (Stage 5b)</span>
        <span class="px-2.5 py-1 text-xs rounded-full font-bold border ${droppedRefs === 0 ? 'bg-emerald-100 text-emerald-800 border-emerald-300' : 'bg-amber-100 text-amber-800 border-amber-300'}">
          ${validRefs}/${totalRefs} Grounded
        </span>
      </div>
      <div class="text-xs text-slate-600">
        ${droppedRefs > 0 ? `<p class="text-amber-800 font-medium">Dropped ${droppedRefs} reference(s) below semantic threshold to prevent hallucinated grounding.</p>` : '<p class="text-emerald-700 font-medium">All retrieved statutory references validated against classified risk categories.</p>'}
      </div>
    `;
  }

  // Stage Latencies
  if (latencyEl) {
    latencyEl.innerHTML = '';
    const latencies = result.stage_latencies_ms || {};
    const total = result.total_latency_ms ? Math.round(result.total_latency_ms) : 0;

    const badges = [
      { name: 'Total', ms: total, color: 'bg-slate-800 text-white' },
      { name: 'Simplifier', ms: latencies.simplifier_ms, color: 'bg-blue-100 text-blue-800' },
      { name: 'Classifier', ms: latencies.classifier_ms, color: 'bg-purple-100 text-purple-800' },
      { name: 'Grounding', ms: latencies.grounding_ms, color: 'bg-amber-100 text-amber-800' },
      { name: 'Verifier', ms: latencies.meaning_verifier_ms, color: 'bg-teal-100 text-teal-800' },
    ];

    badges.forEach(b => {
      if (b.ms !== undefined) {
        const chip = document.createElement('span');
        chip.className = `px-2 py-0.5 rounded text-[11px] font-mono font-medium ${b.color}`;
        chip.textContent = `${b.name}: ${Math.round(b.ms)}ms`;
        latencyEl.appendChild(chip);
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Full Document Analysis
// ---------------------------------------------------------------------------
function initializeDocumentAnalysis() {
  const dropzone = document.getElementById('contract-dropzone');
  const fileInput = document.getElementById('file-upload-input');
  const textInput = document.getElementById('doc-text-input');
  const runBtn = document.getElementById('btn-analyze-document');
  const sampleSelect = document.getElementById('doc-sample-select');
  const loadingSection = document.getElementById('doc-analysis-loading');
  const resultsSection = document.getElementById('doc-analysis-results');

  // Load sample contract
  if (sampleSelect && textInput) {
    sampleSelect.addEventListener('change', (e) => {
      const key = e.target.value;
      if (key && SAMPLE_CONTRACTS[key]) {
        textInput.value = SAMPLE_CONTRACTS[key];
      }
    });
  }

  // Drag and drop handlers
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFileSelection(files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    const filenameLabel = document.getElementById('selected-file-label');
    if (filenameLabel) {
      filenameLabel.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      filenameLabel.classList.remove('hidden');
    }
  }

  // Analyze Document Action
  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      const file = fileInput && fileInput.files.length > 0 ? fileInput.files[0] : null;
      const text = textInput ? textInput.value.trim() : '';

      if (!file && !text) {
        alert('Please select a contract file or paste contract text to analyze.');
        return;
      }

      if (loadingSection) loadingSection.classList.remove('hidden');
      if (resultsSection) resultsSection.classList.add('hidden');
      runBtn.disabled = true;

      try {
        let response;
        if (file) {
          const formData = new FormData();
          formData.append('file', file);
          response = await fetch('/analyze-document/file?max_clauses=50', {
            method: 'POST',
            body: formData,
          });
        } else {
          response = await fetch('/analyze-document', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contract_text: text, max_clauses: 50 }),
          });
        }

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();
        state.documentResults = data;
        renderDocumentResults(data);
      } catch (err) {
        alert(`Document Pipeline Error: ${err.message}\nEnsure LawUP FastAPI service and LM Studio are running.`);
      } finally {
        if (loadingSection) loadingSection.classList.add('hidden');
        if (resultsSection) resultsSection.classList.remove('hidden');
        runBtn.disabled = false;
      }
    });
  }

  // Filter Buttons
  setupFilters();

  // Export Buttons
  setupExports();
}

function renderDocumentResults(data) {
  // Summary Stats
  const totalClausesEl = document.getElementById('stat-total-clauses');
  const flaggedClausesEl = document.getElementById('stat-flagged-clauses');
  const passRateEl = document.getElementById('stat-pass-rate');
  const totalLatencyEl = document.getElementById('stat-total-latency');
  const riskPillsContainer = document.getElementById('summary-risk-pills');

  const clauses = data.clauses || [];
  const riskSummary = data.risk_summary || {};
  const verifSummary = data.verification_summary || {};

  if (totalClausesEl) totalClausesEl.textContent = data.total_clauses || clauses.length;

  let flaggedCount = 0;
  clauses.forEach(c => {
    if (c.risks_identified && c.risks_identified.length > 0) flaggedCount++;
  });
  if (flaggedClausesEl) flaggedClausesEl.textContent = flaggedCount;

  const passRate = verifSummary.pass_rate !== undefined ? `${Math.round(verifSummary.pass_rate * 100)}%` : '100%';
  if (passRateEl) passRateEl.textContent = passRate;

  if (totalLatencyEl) {
    const s = ((data.total_latency_ms || 0) / 1000).toFixed(1);
    totalLatencyEl.textContent = `${s}s`;
  }

  // Risk category pills
  if (riskPillsContainer) {
    riskPillsContainer.innerHTML = '';
    const entries = Object.entries(riskSummary);
    if (entries.length === 0) {
      riskPillsContainer.innerHTML = '<span class="text-xs text-slate-500">No high-risk categories detected.</span>';
    } else {
      entries.forEach(([cat, count]) => {
        const pill = document.createElement('span');
        pill.className = 'px-3 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200 flex items-center space-x-1.5';
        pill.innerHTML = `<span>${escapeHtml(cat)}</span> <span class="bg-rose-200 text-rose-900 rounded-full px-1.5 py-0.2 text-[10px]">${count}</span>`;
        riskPillsContainer.appendChild(pill);
      });
    }
  }

  // Render individual clause cards
  renderClauseCards(clauses);
}

function renderClauseCards(clauses) {
  const container = document.getElementById('clause-cards-container');
  if (!container) return;

  container.innerHTML = '';

  const filtered = clauses.filter(c => {
    // Filter by type
    if (state.activeFilter === 'high') {
      if (!c.risks_identified || c.risks_identified.length === 0) return false;
    } else if (state.activeFilter === 'failed') {
      if (c.verifier_report && c.verifier_report.passes_verification) return false;
    } else if (state.activeFilter === 'clean') {
      if (c.risks_identified && c.risks_identified.length > 0) return false;
    }

    // Search query
    if (state.searchQuery) {
      const q = state.searchQuery.toLowerCase();
      const orig = (c.original_clause || '').toLowerCase();
      const simp = (c.simplified_rewrite || '').toLowerCase();
      const sec = (c.section_heading || '').toLowerCase();
      if (!orig.includes(q) && !simp.includes(q) && !sec.includes(q)) return false;
    }

    return true;
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="text-center py-12 bg-white rounded-xl border border-slate-200">
        <p class="text-sm text-slate-500">No clauses match the current filter or search criteria.</p>
      </div>
    `;
    return;
  }

  filtered.forEach(c => {
    const card = document.createElement('div');
    card.className = 'clause-card bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4';

    const hasRisks = c.risks_identified && c.risks_identified.length > 0;
    const passes = !c.verifier_report || c.verifier_report.passes_verification;

    // Severity indicator
    const severityBadge = hasRisks
      ? '<span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-200">FLAGGED RISK</span>'
      : '<span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">STANDARD CLAUSE</span>';

    // Verifier badge
    const verifBadge = passes
      ? '<span class="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">VERIFIED PASS</span>'
      : '<span class="px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-50 text-amber-800 border border-amber-300">DISCREPANCY DETECTED</span>';

    // Risk tags
    let riskTagsHtml = '';
    if (hasRisks) {
      riskTagsHtml = c.risks_identified
        .map(r => `<span class="px-2 py-0.5 rounded text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200">${escapeHtml(r)}</span>`)
        .join(' ');
    }

    // Grounding references
    let groundingHtml = '';
    if (c.grounding_references && c.grounding_references.length > 0) {
      groundingHtml = `
        <div class="pt-2 border-t border-slate-100">
          <div class="text-[11px] font-semibold uppercase text-slate-400 tracking-wider mb-1.5">Grounded Legal Standards</div>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
            ${c.grounding_references.map(ref => `
              <div class="p-2 rounded bg-slate-50 border border-slate-200 text-xs">
                <div class="flex justify-between items-center mb-1">
                  <span class="font-semibold text-slate-800">${escapeHtml(ref.title)}</span>
                  <span class="text-[10px] font-mono px-1.5 py-0.2 rounded bg-blue-100 text-blue-800">${Math.round((ref.similarity_score || 0) * 100)}%</span>
                </div>
                <div class="text-[11px] text-slate-500">${escapeHtml(ref.authority || 'Legal Authority')}</div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    // Verifier Details
    let verifierDetailsHtml = '';
    if (c.verifier_report && !c.verifier_report.passes_verification) {
      verifierDetailsHtml = `
        <div class="p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 space-y-1">
          <div class="font-bold">Judge Verification Flag: ${escapeHtml(c.verifier_report.failure_category || 'Potential Inconsistency')}</div>
          ${c.verifier_report.explanation ? `<div>${escapeHtml(c.verifier_report.explanation)}</div>` : ''}
        </div>
      `;
    }

    // Latency Telemetry
    const lat = c.stage_latencies_ms || {};
    const latencyHtml = `
      <div class="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500">
        <span class="font-mono text-slate-700 font-medium">Latency: ${Math.round(c.total_latency_ms || 0)}ms</span>
        ${lat.simplifier_ms ? `<span class="bg-slate-100 px-1.5 py-0.5 rounded">Simplifier: ${Math.round(lat.simplifier_ms)}ms</span>` : ''}
        ${lat.classifier_ms ? `<span class="bg-slate-100 px-1.5 py-0.5 rounded">Classifier: ${Math.round(lat.classifier_ms)}ms</span>` : ''}
        ${lat.grounding_ms ? `<span class="bg-slate-100 px-1.5 py-0.5 rounded">Grounding: ${Math.round(lat.grounding_ms)}ms</span>` : ''}
        ${lat.meaning_verifier_ms ? `<span class="bg-slate-100 px-1.5 py-0.5 rounded">Verifier: ${Math.round(lat.meaning_verifier_ms)}ms</span>` : ''}
      </div>
    `;

    card.innerHTML = `
      <div class="flex items-center justify-between pb-2 border-b border-slate-100">
        <div class="flex items-center space-x-2">
          <span class="font-bold text-slate-800 text-sm">Clause #${c.clause_index + 1}</span>
          ${c.section_heading ? `<span class="text-xs text-slate-500 font-medium">&bull; ${escapeHtml(c.section_heading)}</span>` : ''}
        </div>
        <div class="flex items-center space-x-2">
          ${severityBadge}
          ${verifBadge}
        </div>
      </div>

      ${hasRisks ? `<div class="flex flex-wrap gap-1.5">${riskTagsHtml}</div>` : ''}

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
        <div class="space-y-1">
          <div class="text-[11px] font-semibold uppercase text-slate-400 tracking-wider">Original Contract Legalese</div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-700 leading-relaxed max-h-48 overflow-y-auto">
            ${escapeHtml(c.original_clause)}
          </div>
        </div>
        <div class="space-y-1">
          <div class="text-[11px] font-semibold uppercase text-slate-400 tracking-wider text-blue-600">Plain-English Simplification</div>
          <div class="p-3 bg-blue-50/60 rounded-lg border border-blue-200 text-xs text-slate-800 leading-relaxed max-h-48 overflow-y-auto">
            ${escapeHtml(c.simplified_rewrite || 'Simplification omitted.')}
          </div>
        </div>
      </div>

      ${verifierDetailsHtml}
      ${groundingHtml}
      ${latencyHtml}
    `;

    container.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Filtering & Exports
// ---------------------------------------------------------------------------
function setupFilters() {
  const filterButtons = document.querySelectorAll('.filter-btn');
  const searchInput = document.getElementById('search-clauses-input');

  filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const filter = btn.getAttribute('data-filter');
      state.activeFilter = filter;

      filterButtons.forEach(b => b.classList.remove('bg-blue-600', 'text-white'));
      filterButtons.forEach(b => b.classList.add('bg-white', 'text-slate-600'));
      btn.classList.remove('bg-white', 'text-slate-600');
      btn.classList.add('bg-blue-600', 'text-white');

      if (state.documentResults) {
        renderClauseCards(state.documentResults.clauses || []);
      }
    });
  });

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      state.searchQuery = e.target.value.trim();
      if (state.documentResults) {
        renderClauseCards(state.documentResults.clauses || []);
      }
    });
  }
}

function setupExports() {
  const exportJsonBtn = document.getElementById('btn-export-json');
  const exportMdBtn = document.getElementById('btn-export-markdown');

  if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', () => {
      if (!state.documentResults) return;
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.documentResults, null, 2));
      const dlAnchor = document.createElement('a');
      dlAnchor.setAttribute('href', dataStr);
      dlAnchor.setAttribute('download', `lawup-contract-analysis-${Date.now()}.json`);
      dlAnchor.click();
    });
  }

  if (exportMdBtn) {
    exportMdBtn.addEventListener('click', () => {
      if (!state.documentResults) return;
      const md = generateMarkdownReport(state.documentResults);
      const dataStr = 'data:text/markdown;charset=utf-8,' + encodeURIComponent(md);
      const dlAnchor = document.createElement('a');
      dlAnchor.setAttribute('href', dataStr);
      dlAnchor.setAttribute('download', `lawup-contract-report-${Date.now()}.md`);
      dlAnchor.click();
    });
  }
}

function generateMarkdownReport(data) {
  const clauses = data.clauses || [];
  const riskSummary = data.risk_summary || {};
  const verifSummary = data.verification_summary || {};

  let md = `# LawUP Contract Risk Analysis Report\n`;
  md += `Generated: ${new Date().toISOString()}\n\n`;
  md += `## Executive Summary\n`;
  md += `- **Total Clauses Analyzed:** ${data.total_clauses || clauses.length}\n`;
  md += `- **Total Processing Latency:** ${((data.total_latency_ms || 0) / 1000).toFixed(1)}s\n`;
  md += `- **NLI Verification Pass Rate:** ${Math.round((verifSummary.pass_rate || 1.0) * 100)}%\n\n`;

  md += `### Risk Distribution\n`;
  for (const [k, v] of Object.entries(riskSummary)) {
    md += `- **${k}:** ${v} occurrence(s)\n`;
  }
  md += `\n---\n\n## Clause Analysis\n\n`;

  clauses.forEach(c => {
    md += `### Clause #${c.clause_index + 1}: ${c.section_heading || 'Untitled'}\n\n`;
    if (c.risks_identified && c.risks_identified.length > 0) {
      md += `**Identified Risks:** ${c.risks_identified.join(', ')}\n\n`;
    }
    md += `**Original Text:**\n> ${c.original_clause.replace(/\n/g, ' ')}\n\n`;
    md += `**Plain-English Simplification:**\n> ${(c.simplified_rewrite || '').replace(/\n/g, ' ')}\n\n`;

    if (c.grounding_references && c.grounding_references.length > 0) {
      md += `**Statutory Citations:**\n`;
      c.grounding_references.forEach(ref => {
        md += `- ${ref.title} (${ref.authority}) - Match: ${Math.round((ref.similarity_score || 0) * 100)}%\n`;
      });
      md += `\n`;
    }

    if (c.verifier_report && !c.verifier_report.passes_verification) {
      md += `> [!WARNING]\n> **Discrepancy Detected:** ${c.verifier_report.failure_category}\n> ${c.verifier_report.explanation || ''}\n\n`;
    }
    md += `---\n\n`;
  });

  return md;
}

// Utility: HTML Escaping
function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
