async function runPrediction() {
    const teamA = document.getElementById('teamA').value.trim();
    const teamB = document.getElementById('teamB').value.trim();
    const btn = document.getElementById('predictBtn');
    const resultsSection = document.getElementById('resultsSection');
    const errorMsg = document.getElementById('errorMsg');

    if (!teamA || !teamB) {
        alert("Ambos equipos son requeridos.");
        return;
    }

    btn.classList.add('loading');
    resultsSection.classList.remove('hidden');
    errorMsg.classList.add('hidden');
    
    // Reset values a 0 para efecto visual
    animateValue(document.querySelector('#probHome .prob-value'), 0, 0, 0);
    animateValue(document.querySelector('#probDraw .prob-value'), 0, 0, 0);
    animateValue(document.querySelector('#probAway .prob-value'), 0, 0, 0);

    try {
        const response = await fetch('http://localhost:8000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team_a: teamA, team_b: teamB })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Error en el motor");
        }

        const data = await response.json();
        
        // Actualizar Probabilidades 1X2
        animateValue(document.querySelector('#probHome .prob-value'), 0, data.p1 * 100, 1000);
        animateValue(document.querySelector('#probDraw .prob-value'), 0, data.px * 100, 1000);
        animateValue(document.querySelector('#probAway .prob-value'), 0, data.p2 * 100, 1000);

        // Actualizar Lambdas
        document.getElementById('lambdaA').innerText = data.lam.toFixed(2);
        document.getElementById('lambdaB').innerText = data.mu.toFixed(2);

        // Actualizar Top 5 Scores
        const topScoresDiv = document.getElementById('topScores');
        topScoresDiv.innerHTML = '';
        const maxProb = data.top_5_scores[0].prob;

        data.top_5_scores.forEach(score => {
            const row = document.createElement('div');
            row.className = 'score-row';
            row.innerHTML = `
                <div class="score-val">${score.goals_a} - ${score.goals_b}</div>
                <div style="display:flex; align-items:center; flex:1; justify-content:flex-end;">
                    <div class="score-bar"><div class="score-fill" style="width: ${(score.prob/maxProb)*100}%"></div></div>
                    <span class="score-prob" style="margin-left:15px; width:60px; text-align:right;">${(score.prob*100).toFixed(1)}%</span>
                </div>
            `;
            topScoresDiv.appendChild(row);
        });

    } catch (error) {
        console.error("Error:", error);
        errorMsg.innerText = `ERROR DEL SISTEMA: ${error.message}`;
        errorMsg.classList.remove('hidden');
    } finally {
        btn.classList.remove('loading');
    }
}

// Función para animar contadores de números
function animateValue(element, start, end, duration) {
    const range = end - start;
    const minTimer = 50;
    const stepTime = Math.max(Math.abs(Math.floor(duration / range)), minTimer);
    const startTime = new Date().getTime();
    const endTime = startTime + duration;
    
    function run() {
        const now = new Date().getTime();
        const remaining = Math.max((endTime - now) / duration, 0);
        const value = Math.round(end - (remaining * range));
        element.innerText = value.toFixed(1) + '%';
        if (value === end) return;
        setTimeout(run, stepTime);
    }
    if (range === 0) {
        element.innerText = '0.0%';
    } else {
        run();
    }
}
