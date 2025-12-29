from flask import Flask, request, send_file, render_template, jsonify
from infer import generate_from_input

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/generate", methods=["POST"])
def gen():
    data = request.json or {}
    # Expect frontend to send attrs = [Smiling, Male] and expression as hint
    attrs = data.get("attrs")
    expression = data.get("expression")
    variations = int(data.get("variations", 1))

    if attrs is None:
        # Fallback: derive from gender + expression if frontend didn't send attrs
        gender = data.get('gender', 'female')
        male = 1 if str(gender).lower() == 'male' else 0
        smiling = 1 if str(expression or '').lower() == 'happy' else 0
        attrs = [smiling, male]

    # Use attrs exactly as provided for the model; expression is only a latent hint
    images = generate_from_input(attrs, variations=variations, expression=expression)
    return jsonify({"images": images})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
