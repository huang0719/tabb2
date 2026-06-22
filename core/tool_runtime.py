from pathlib import Path
from typing import Any
import re


# // 判断路径是否位于允许目录内
def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# // 获取工具允许写入目录
def _allowed_roots(config) -> list[Path]:
    configured = config.get("tools", "allowed_roots", default=None) if config else None
    if configured:
        return [Path(item) for item in configured]
    repo_root = Path(__file__).resolve().parent.parent
    return [repo_root, Path("C:\\")]


# // 校验并解析写入路径
def _resolve_write_path(raw_path: str, config) -> Path:
    if not raw_path:
        raise Exception("WriteFile missing path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    resolved = path.resolve()
    if not any(_is_inside(resolved, root) for root in _allowed_roots(config)):
        raise Exception(f"WriteFile path not allowed: {resolved}")
    return resolved


# // 执行 WriteFile 类工具
def _execute_write_file(arguments: dict[str, Any], config) -> dict[str, Any]:
    raw_path = arguments.get("path") or arguments.get("file_path") or arguments.get("filepath")
    content = arguments.get("content")
    if content is None:
        raise Exception("WriteFile missing content")
    target = _resolve_write_path(str(raw_path), config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8", newline="")
    return {
        "ok": True,
        "tool": "WriteFile",
        "path": str(target),
        "bytes": len(str(content).encode("utf-8")),
    }


# // 执行本地工具调用
def execute_tool_call(name: str, arguments: dict[str, Any], config=None) -> dict[str, Any]:
    normalized = name.lower()
    if normalized in ("writefile", "write_file", "create_file"):
        return _execute_write_file(arguments, config)
    raise Exception(f"unsupported tool: {name}")


# // 判断是否为写文件工具
def is_write_file_tool(name: str) -> bool:
    return name.lower() in ("writefile", "write_file", "create_file")


# // 从用户文本中提取写入路径
def extract_write_path(user_text: str) -> str | None:
    match = re.search(r"([A-Za-z]:\\[\s\S]+?)(?:[，,。；;]|\s*$)", user_text)
    if match:
        return match.group(1).strip()
    match = re.search(r"(?:写入|保存到|创建文件)\s*([^\s，。；;]+)", user_text)
    if match:
        return match.group(1).strip()
    return None


# // 构建文件内容生成提示词
def build_file_generation_prompt(user_text: str, target_path: str) -> str:
    suffix = Path(target_path).suffix.lower()
    if suffix in (".html", ".htm"):
        return (
            "根据用户需求生成一个完整、可直接运行的单文件 HTML。\n"
            "只输出文件内容，不要解释，不要 Markdown，不要代码围栏。\n"
            "HTML 必须包含 <!doctype html>、<html>、<head>、<body>，所有标签必须闭合。\n"
            "如果需要外部库，使用普通 <script src=\"...\"></script> 标签，禁止 Markdown 链接。\n"
            f"目标文件: {target_path}\n"
            f"用户需求: {user_text}"
        )
    return (
        "根据用户需求生成目标文件的完整内容。\n"
        "只输出文件内容，不要解释，不要 Markdown，不要代码围栏。\n"
        f"目标文件: {target_path}\n"
        f"用户需求: {user_text}"
    )


# // 从模型输出中提取文件内容
def extract_generated_file_content(text: str) -> str:
    fence = re.search(r"```(?:html|javascript|js|css|python|txt)?\s*\n([\s\S]*?)\n?```", text)
    if fence:
        return fence.group(1).strip()
    html_start = re.search(r"<!doctype html|<html[\s>]", text, re.I)
    if html_start:
        return text[html_start.start() :].strip()
    return text.strip()


# // 生成内置 3D 魔方 HTML
def build_builtin_file_content(user_text: str, target_path: str) -> str:
    suffix = Path(target_path).suffix.lower()
    lowered = user_text.lower()
    if suffix not in (".html", ".htm"):
        return ""
    if "魔方" not in user_text and "rubik" not in lowered and "cube" not in lowered:
        return ""
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3D 可玩魔方</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #101827; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
    #panel { position: fixed; left: 16px; top: 16px; z-index: 10; color: #fff; background: rgba(15,23,42,.82); border: 1px solid rgba(255,255,255,.14); border-radius: 10px; padding: 12px; max-width: 340px; }
    h1 { margin: 0 0 8px; font-size: 16px; }
    p { margin: 4px 0 10px; font-size: 12px; color: #cbd5e1; }
    .grid { display: grid; grid-template-columns: repeat(6, 44px); gap: 8px; }
    button { height: 34px; border: 0; border-radius: 8px; color: white; background: #2563eb; font-weight: 700; cursor: pointer; }
    button.secondary { background: #475569; }
    button:active { transform: translateY(1px); }
    #status { margin-top: 10px; min-height: 18px; color: #93c5fd; font-size: 12px; }
  </style>
</head>
<body>
  <div id="panel">
    <h1>3D 可玩魔方</h1>
    <p>拖拽旋转视角，滚轮缩放。点击按钮转动对应层。</p>
    <div class="grid">
      <button data-move="U">U</button><button data-move="R">R</button><button data-move="F">F</button><button data-move="L">L</button><button data-move="D">D</button><button data-move="B">B</button>
      <button data-move="U'">U'</button><button data-move="R'">R'</button><button data-move="F'">F'</button><button data-move="L'">L'</button><button data-move="D'">D'</button><button data-move="B'">B'</button>
    </div>
    <div class="grid" style="grid-template-columns: repeat(3, 92px); margin-top: 10px;">
      <button id="scramble">打乱</button><button id="undo" class="secondary">撤销</button><button id="reset" class="secondary">重置</button>
    </div>
    <div id="status">Ready</div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
    camera.position.set(6, 5, 7);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    const light = new THREE.DirectionalLight(0xffffff, 1.1);
    light.position.set(5, 8, 6);
    scene.add(light);

    const colors = { px: 0xf43f5e, nx: 0xf97316, py: 0xffffff, ny: 0xfacc15, pz: 0x22c55e, nz: 0x3b82f6, inner: 0x111827 };
    const cubies = [];
    const history = [];
    let busy = false;

    function roundedBox(color) {
      const group = new THREE.Group();
      const box = new THREE.BoxGeometry(.96, .96, .96);
      const mats = [colors.px, colors.nx, colors.py, colors.ny, colors.pz, colors.nz].map(c => new THREE.MeshStandardMaterial({ color: c, roughness: .55 }));
      group.add(new THREE.Mesh(box, mats));
      const edges = new THREE.EdgesGeometry(box);
      group.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x020617, linewidth: 2 })));
      return group;
    }

    function makeCube() {
      cubies.length = 0;
      for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
        const c = roundedBox();
        c.position.set(x * 1.05, y * 1.05, z * 1.05);
        c.userData.coord = { x, y, z };
        scene.add(c);
        cubies.push(c);
      }
    }

    function layerFor(move) {
      const m = move[0], prime = move.includes("'");
      const spec = {
        U: ['y', 1, 1], D: ['y', -1, -1], R: ['x', 1, -1], L: ['x', -1, 1], F: ['z', 1, -1], B: ['z', -1, 1]
      }[m];
      return { axis: spec[0], value: spec[1], dir: prime ? -spec[2] : spec[2] };
    }

    function rotateCoord(coord, axis, dir) {
      let { x, y, z } = coord;
      if (axis === 'x') [y, z] = dir > 0 ? [-z, y] : [z, -y];
      if (axis === 'y') [x, z] = dir > 0 ? [z, -x] : [-z, x];
      if (axis === 'z') [x, y] = dir > 0 ? [-y, x] : [y, -x];
      return { x, y, z };
    }

    function turn(move, record = true) {
      if (busy) return;
      busy = true;
      const { axis, value, dir } = layerFor(move);
      const group = new THREE.Group();
      scene.add(group);
      const selected = cubies.filter(c => c.userData.coord[axis] === value);
      selected.forEach(c => group.attach(c));
      const start = performance.now(), duration = 240, angle = dir * Math.PI / 2;
      function step(now) {
        const t = Math.min(1, (now - start) / duration);
        group.rotation[axis] = angle * (1 - Math.pow(1 - t, 3));
        if (t < 1) requestAnimationFrame(step);
        else {
          selected.forEach(c => {
            c.userData.coord = rotateCoord(c.userData.coord, axis, dir);
            scene.attach(c);
            c.position.set(c.userData.coord.x * 1.05, c.userData.coord.y * 1.05, c.userData.coord.z * 1.05);
            c.rotation.x = Math.round(c.rotation.x / (Math.PI / 2)) * Math.PI / 2;
            c.rotation.y = Math.round(c.rotation.y / (Math.PI / 2)) * Math.PI / 2;
            c.rotation.z = Math.round(c.rotation.z / (Math.PI / 2)) * Math.PI / 2;
          });
          scene.remove(group);
          if (record) history.push(move);
          document.getElementById('status').textContent = move;
          busy = false;
        }
      }
      requestAnimationFrame(step);
    }

    function inverse(move) { return move.includes("'") ? move[0] : move + "'"; }
    document.querySelectorAll('[data-move]').forEach(b => b.onclick = () => turn(b.dataset.move));
    document.getElementById('undo').onclick = () => { const m = history.pop(); if (m) turn(inverse(m), false); };
    document.getElementById('reset').onclick = () => { cubies.forEach(c => scene.remove(c)); history.length = 0; makeCube(); };
    document.getElementById('scramble').onclick = () => {
      const moves = ['U','R','F','L','D','B',"U'","R'","F'","L'","D'","B'"];
      let i = 0; const run = () => { if (i++ < 24) { turn(moves[Math.floor(Math.random() * moves.length)]); setTimeout(run, 280); } };
      run();
    };
    addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
    makeCube();
    (function animate(){ controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate); })();
  </script>
</body>
</html>"""


# // 格式化工具执行结果
def format_tool_result_message(result: dict[str, Any]) -> str:
    tool = str(result.get("tool", ""))
    if tool == "WriteFile":
        path = result.get("path")
        size = result.get("bytes")
        return f"已写入文件：{path}（{size} bytes）。"
    return f"工具执行完成：{result}"
