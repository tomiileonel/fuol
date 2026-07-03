const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = `${protocol}//${window.location.host}/ws`;
let ws;

const statusIndicator = document.getElementById("status-indicator");
const alphaVal = document.getElementById("alpha-val");
const kellyVal = document.getElementById("kelly-val");
const engineProb = document.getElementById("engine-prob");
const marketProb = document.getElementById("market-prob");
const tickVal = document.getElementById("tick-val");

const signalPanel = document.getElementById("signal-panel");
const kellyPanel = document.getElementById("kelly-panel");

function connectWebSocket() {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        statusIndicator.innerHTML = '<span class="pulse"></span> STREAM CONECTADO';
        statusIndicator.style.color = "var(--accent-positive)";
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Update raw values
        tickVal.innerText = data.tick;
        engineProb.innerText = (data.engine_prob * 100).toFixed(2) + "%";
        marketProb.innerText = (data.market_prob * 100).toFixed(2) + "%";
        
        // Format Alpha
        const alphaPct = data.alpha * 100;
        alphaVal.innerText = alphaPct > 0 ? `+${alphaPct.toFixed(2)}%` : `${alphaPct.toFixed(2)}%`;
        
        // Format Kelly
        const kellyPct = data.kelly * 100;
        kellyVal.innerText = `${kellyPct.toFixed(2)}%`;
        
        // Styling based on status and alpha
        if (data.status === "DANGER") {
            alphaVal.className = "value danger";
            alphaVal.innerText = "ANOMALÍA (PAUSED)";
            signalPanel.style.borderColor = "var(--accent-negative)";
            kellyVal.className = "value negative";
            kellyVal.innerText = "0.00%";
        } else {
            signalPanel.style.borderColor = "var(--border-color)";
            if (alphaPct > 3) {
                // Value Bet Detected (Green)
                alphaVal.className = "value positive";
                kellyVal.className = "value positive";
            } else {
                alphaVal.className = "value negative";
                kellyVal.className = "value negative";
            }
        }
    };

    ws.onclose = () => {
        statusIndicator.innerHTML = '<span class="pulse" style="background-color: var(--accent-negative); box-shadow: none; animation: none;"></span> DESCONECTADO';
        statusIndicator.style.color = "var(--accent-negative)";
        alphaVal.innerText = "--";
        alphaVal.className = "value negative";
        kellyVal.innerText = "--";
        kellyVal.className = "value negative";
        
        setTimeout(connectWebSocket, 3000); // Auto-reconnect
    };
}

connectWebSocket();
