from __future__ import annotations

import webbrowser
from http.server import SimpleHTTPRequestHandler, HTTPServer
import socketserver
import threading


HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>EMNIST Letter Predictor</title>
    <style>
      body { font-family: sans-serif; margin: 24px; }
      .row { display: flex; gap: 16px; align-items: center; }
      #canvas { border: 1px solid #ccc; background: #fff; touch-action: none; }
      .controls { display: flex; gap: 8px; margin-top: 8px; }
      button { padding: 8px 12px; }
      .result { margin-top: 16px; font-size: 18px; }
    </style>
  </head>
  <body>
    <h2>Draw a letter (A-Z)</h2>
    <div class="row">
      <canvas id="canvas" width="280" height="280"></canvas>
      <div>
        <div class="controls">
          <button id="clear">Clear</button>
          <button id="predict">Predict</button>
        </div>
        <div class="result" id="result">Prediction: -</div>
      </div>
    </div>
    <script>
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 20;
      ctx.lineCap = 'round';

      let drawing = false;
      let last = null;

      const getPos = (e) => {
        if (e.touches && e.touches.length) {
          const rect = canvas.getBoundingClientRect();
          return {
            x: e.touches[0].clientX - rect.left,
            y: e.touches[0].clientY - rect.top,
          };
        }
        const rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
      };

      const start = (e) => { drawing = true; last = getPos(e); };
      const end = () => { drawing = false; last = null; };
      const move = (e) => {
        if (!drawing) return;
        const p = getPos(e);
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
        last = p;
      };

      canvas.addEventListener('mousedown', start);
      canvas.addEventListener('mousemove', move);
      canvas.addEventListener('mouseup', end);
      canvas.addEventListener('mouseleave', end);
      canvas.addEventListener('touchstart', (e) => { e.preventDefault(); start(e); });
      canvas.addEventListener('touchmove', (e) => { e.preventDefault(); move(e); });
      canvas.addEventListener('touchend', (e) => { e.preventDefault(); end(); });

      document.getElementById('clear').onclick = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        document.getElementById('result').textContent = 'Prediction: -';
      };

      document.getElementById('predict').onclick = async () => {
        const dataUrl = canvas.toDataURL('image/png');
        document.getElementById('result').textContent = 'Predicting...';
        try {
          const resp = await fetch('http://localhost:5001/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
          });
          const json = await resp.json();
          if (!resp.ok) throw new Error(json.error || 'Request failed');
          const { prediction, confidence } = json;
          document.getElementById('result').textContent = `Prediction: ${prediction} (conf ${(confidence*100).toFixed(1)}%)`;
        } catch (err) {
          document.getElementById('result').textContent = 'Error: ' + err.message;
        }
      };
    </script>
  </body>
  </html>
"""


class _Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            content = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        return super().do_GET()


def serve(host: str = '127.0.0.1', port: int = 8000, open_browser: bool = True) -> None:
    httpd = HTTPServer((host, port), _Handler)
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(f"http://{host}:{port}"), daemon=True).start()
    print(f"Serving frontend at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()

