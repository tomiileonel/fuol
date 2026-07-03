document.addEventListener("DOMContentLoaded", () => {
    const loadBtn = document.getElementById("loadBtn");
    const matchIdInput = document.getElementById("matchIdInput");
    
    const content = document.getElementById("content");
    const loading = document.getElementById("loading");
    const errorState = document.getElementById("error");
    const errorMsg = document.getElementById("errorMsg");

    async function fetchPrediction(matchId) {
        // Reset states
        content.classList.add("hidden");
        errorState.classList.add("hidden");
        loading.classList.remove("hidden");

        try {
            const response = await fetch(`/api/predictions/${matchId}`);
            if (!response.ok) {
                throw new Error(response.status === 404 ? "Predicción no encontrada en el Data Lake." : "Error de conexión con el servidor.");
            }
            
            const data = await response.json();
            populateDashboard(data);
            
            loading.classList.add("hidden");
            content.classList.remove("hidden");
            
        } catch (error) {
            loading.classList.add("hidden");
            errorMsg.textContent = error.message;
            errorState.classList.remove("hidden");
        }
    }

    function formatCurrency(val) {
        if (!val) return "N/A";
        if (val >= 1e9) return `€${(val / 1e9).toFixed(2)}B`;
        if (val >= 1e6) return `€${(val / 1e6).toFixed(2)}M`;
        return `€${val}`;
    }
    
    function formatProb(prob) {
        if (prob === undefined || prob === null) return "N/A";
        return `${(prob * 100).toFixed(1)}%`;
    }

    function populateDashboard(data) {
        // Team names from match_id (e.g. Argentina_vs_France)
        const teams = data.match_id.split("_vs_");
        const teamAName = teams[0] || "Team A";
        const teamBName = teams[1] || "Team B";

        // Headers
        document.querySelector("#teamA .team-name").textContent = teamAName;
        document.querySelector("#teamB .team-name").textContent = teamBName;
        document.querySelector("#teamA .coach span").textContent = data.metadata?.team_a_info?.coach || "-";
        document.querySelector("#teamB .coach span").textContent = data.metadata?.team_b_info?.coach || "-";

        // Finance
        document.getElementById("teamANameFin").textContent = teamAName;
        document.getElementById("teamBNameFin").textContent = teamBName;
        document.getElementById("valA").textContent = formatCurrency(data.metadata?.team_a_info?.value_eur);
        document.getElementById("valB").textContent = formatCurrency(data.metadata?.team_b_info?.value_eur);

        // NLP Modifiers
        document.getElementById("teamANameNLP").textContent = teamAName;
        document.getElementById("teamBNameNLP").textContent = teamBName;
        
        const mods = data.web_context?.extracted_modifiers || {};
        document.getElementById("injA").textContent = (mods.team_a?.injury_modifier || 1.0).toFixed(2) + "x";
        document.getElementById("fatA").textContent = (mods.team_a?.travel_fatigue || 1.0).toFixed(2) + "x";
        document.getElementById("injB").textContent = (mods.team_b?.injury_modifier || 1.0).toFixed(2) + "x";
        document.getElementById("fatB").textContent = (mods.team_b?.travel_fatigue || 1.0).toFixed(2) + "x";

        // Engine Prediction
        const pred = data.engine_prediction || {};
        window.currentEnginePrediction = pred; // Guardar para Paper Trading
        
        document.getElementById("p1Value").textContent = formatProb(pred.p1);
        document.getElementById("pxValue").textContent = formatProb(pred.px);
        document.getElementById("p2Value").textContent = formatProb(pred.p2);
        
        document.getElementById("lamValue").textContent = pred.lam ? pred.lam.toFixed(2) : "-";
        document.getElementById("muValue").textContent = pred.mu ? pred.mu.toFixed(2) : "-";
        
        // Refresh Portfolio
        fetchPortfolio();
    }

    async function fetchPortfolio() {
        try {
            const response = await fetch("/api/portfolio");
            if (response.ok) {
                const data = await response.json();
                document.getElementById("bankrollValue").textContent = `$${data.current_bankroll.toFixed(2)}`;
                document.getElementById("roiValue").textContent = `${data.roi_percent.toFixed(2)}%`;
                
                // Render Trade History
                const tbody = document.getElementById("tradeHistoryBody");
                tbody.innerHTML = "";
                data.history.forEach(trade => {
                    const tr = document.createElement("tr");
                    
                    let statusClass = "status-pending";
                    if (trade.status === "WON") statusClass = "status-won";
                    if (trade.status === "LOST") statusClass = "status-lost";
                    if (trade.status === "VOID") statusClass = "status-void";
                    
                    tr.innerHTML = `
                        <td class="mono">${trade.match_id}</td>
                        <td>${trade.selection}</td>
                        <td class="mono">$${trade.stake.toFixed(2)}</td>
                        <td class="${statusClass}">${trade.status}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        } catch (error) {
            console.error("Error fetching portfolio:", error);
        }
    }

    const execTradeBtn = document.getElementById("execTradeBtn");
    const tradeFeedback = document.getElementById("tradeFeedback");

    execTradeBtn.addEventListener("click", async () => {
        const matchId = matchIdInput.value.trim();
        const selection = document.getElementById("tradeSelection").value;
        const oddsStr = document.getElementById("tradeOdds").value;
        const odds = parseFloat(oddsStr);

        if (!matchId || !window.currentEnginePrediction || !oddsStr || isNaN(odds) || odds <= 1) {
            showFeedback("Datos inválidos para operar.", false);
            return;
        }

        // Determinar probabilidad base del motor según selección
        let engProb = 0;
        if (selection === "1") engProb = window.currentEnginePrediction.p1;
        else if (selection === "X") engProb = window.currentEnginePrediction.px;
        else if (selection === "2") engProb = window.currentEnginePrediction.p2;

        execTradeBtn.disabled = true;
        execTradeBtn.textContent = "PROCESANDO...";
        
        try {
            const response = await fetch(`/api/paper_trade/${matchId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    selection: selection,
                    engine_prob: engProb,
                    market_odds: odds
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                showFeedback(data.detail || "Error al ejecutar orden.", false);
            } else {
                showFeedback(`Orden EJECUTADA. Stake: $${data.trade.stake.toFixed(2)}`, true);
                fetchPortfolio(); // Refrescar Bankroll
            }
        } catch (error) {
            showFeedback("Fallo de red al operar.", false);
        } finally {
            execTradeBtn.disabled = false;
            execTradeBtn.textContent = "EJECUTAR ORDEN";
        }
    });

    function showFeedback(msg, isSuccess) {
        tradeFeedback.textContent = msg;
        tradeFeedback.className = isSuccess ? "feedback-success" : "feedback-error";
        tradeFeedback.classList.remove("hidden");
        setTimeout(() => {
            tradeFeedback.classList.add("hidden");
        }, 5000);
    }

    loadBtn.addEventListener("click", () => {
        const matchId = matchIdInput.value.trim();
        if (matchId) {
            fetchPrediction(matchId);
        }
    });

    // Allow enter key
    matchIdInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            loadBtn.click();
        }
    });
    
    // Initial fetch if value exists
    if(matchIdInput.value.trim()){
        fetchPrediction(matchIdInput.value.trim());
    }
});
