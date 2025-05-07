const questionInput = document.getElementById('questionInput');
const askButton = document.getElementById('askButton');
const responseArea = document.getElementById('responseArea');
const contextArea = document.getElementById('contextArea');

const API_URL = 'http://127.0.0.1:8000/ask';

function toggleFoldable(header) {
    const content = header.nextElementSibling;
    content.classList.toggle('collapsed');
    const arrow = header.querySelector('h3');
    if (content.classList.contains('collapsed')) {
        arrow.textContent = arrow.textContent.replace('▼', '▶');
    } else {
        arrow.textContent = arrow.textContent.replace('▶', '▼');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const foldableHeaders = document.querySelectorAll('.foldable-header');
    foldableHeaders.forEach(header => {
        const content = header.nextElementSibling;
        content.classList.add('collapsed');
    });
});


askButton.addEventListener('click', async () => {
    const question = questionInput.value.trim();

    if (!question) {
        responseArea.textContent = 'Per favore, inserisci una domanda.';
        contextArea.textContent = 'Nessun contesto recuperato.';
        return; // Esci dalla funzione
    }

    responseArea.textContent = 'Generazione risposta in corso...';
    contextArea.textContent = 'Recupero contesto in corso...';
    askButton.disabled = true;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({question: question}),
        });

        if (!response.ok) {
            new Error(`Errore HTTP: ${response.status}`);
        }

        const data = await response.json();

        responseArea.textContent = data.answer;

        if (data.retrieved_contexts && data.retrieved_contexts.length > 0) {
            contextArea.textContent = data.retrieved_contexts.join('\n---\n');
        } else {
            contextArea.textContent = 'Nessun contesto rilevante recuperato.';
        }

    } catch (error) {
        console.error('Errore durante la richiesta:', error);
        responseArea.textContent = `Errore: Impossibile connettersi al server o elaborare la richiesta. (${error.message})`;
        contextArea.textContent = 'Errore nel recupero del contesto.';
    } finally {
        askButton.disabled = false;
    }
});
