from pathlib import Path
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path("site")
output.mkdir(exist_ok=True)

now = datetime.now(timezone.utc)
time_label = now.strftime("%d %B %Y, %H:%M UTC")

plt.figure(figsize=(10, 5))
plt.plot([0, 1, 2, 3, 4], [5, 8, 6, 11, 9], linewidth=2)
plt.title("Automatically generated: " + time_label)
plt.xlabel("Time")
plt.ylabel("Test value")
plt.grid(alpha=0.2)
plt.savefig(output / "test.png", dpi=150)
plt.close()

html = f"""
<html>
<head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family:Arial;text-align:center;padding:20px">
<h1>My automatic weather maps</h1>
<p>Last update: {time_label}</p>
<img src="test.png?v={now.timestamp()}" style="max-width:100%">
</body>
</html>
"""

(output / "index.html").write_text(html, encoding="utf-8")
