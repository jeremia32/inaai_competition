const messagesEl = document.getElementById('messages')
const queryInput = document.getElementById('query')
const sendBtn = document.getElementById('send')

function appendMessage(text, cls='bot'){
  const el = document.createElement('div')
  el.className = 'msg ' + cls
  el.innerHTML = text
  messagesEl.appendChild(el)
  messagesEl.scrollTop = messagesEl.scrollHeight
}

async function sendQuery(){
  const q = queryInput.value.trim()
  if(!q) return
  appendMessage(escapeHtml(q), 'user')
  queryInput.value = ''

  appendMessage('Mencari jawaban…', 'bot')

  try{
    const res = await fetch('/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, top_k: 5 }),
    })

    if(!res.ok){
      const text = await res.text()
      appendMessage('Error: ' + escapeHtml(text), 'bot')
      return
    }

    const data = await res.json()
    // remove the 'loading' message
    const last = messagesEl.querySelector('.bot:last-child')
    if(last && last.textContent.includes('Mencari jawaban')) last.remove()

    appendMessage('<strong>Jawaban:</strong><br/>' + nl2br(escapeHtml(data.answer || '')), 'bot')

    if(data.sources && data.sources.length){
      let srcHtml = '<div class="sources"><strong>Sumber:</strong><ul>'
      data.sources.forEach(s => {
        srcHtml += `<li>[${s.rank}] ${escapeHtml(s.metadata.file_name || 'file')} (hal ${s.metadata.page || '?'})</li>`
      })
      srcHtml += '</ul></div>'
      appendMessage(srcHtml, 'bot')
    }

  }catch(err){
    appendMessage('Error fetching API: ' + escapeHtml(err.message), 'bot')
  }
}

function nl2br(str){ return str.replace(/\n/g, '<br/>') }
function escapeHtml(unsafe){
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

sendBtn.addEventListener('click', sendQuery)
queryInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ sendQuery() }})

// show a welcome message
appendMessage('<strong>Selamat datang</strong> — tanyakan masalah medis berbasis dokumen.','bot')
