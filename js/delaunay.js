// Minimal Bowyer–Watson Delaunay triangulation. Runs once on ~468 points at
// startup, so the O(n^2) simplicity is preferable to pulling in a dependency.

/**
 * @param {Array<[number, number]>} points
 * @returns {Uint16Array} flat list of triangle vertex indices (3 per triangle)
 */
export function triangulate(points) {
  const n = points.length;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  const d = Math.max(maxX - minX, maxY - minY) * 20;
  const mx = (minX + maxX) / 2;
  const my = (minY + maxY) / 2;
  const pts = points.concat([[mx - d, my - d], [mx + d, my - d], [mx, my + d]]);

  const circum = (a, b, c) => {
    const [ax, ay] = pts[a], [bx, by] = pts[b], [cx, cy] = pts[c];
    const den = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (Math.abs(den) < 1e-12) return { a, b, c, x: 0, y: 0, r2: Infinity };
    const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
    const x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / den;
    const y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / den;
    return { a, b, c, x, y, r2: (ax - x) ** 2 + (ay - y) ** 2 };
  };

  let tris = [circum(n, n + 1, n + 2)];
  for (let i = 0; i < n; i++) {
    const [px, py] = pts[i];
    const keep = [];
    const edges = new Map();
    const addEdge = (u, v) => {
      const key = u < v ? u * 65536 + v : v * 65536 + u;
      if (edges.has(key)) edges.delete(key); // shared edge: interior of the cavity
      else edges.set(key, [u, v]);
    };
    for (const t of tris) {
      if ((px - t.x) ** 2 + (py - t.y) ** 2 <= t.r2 * (1 + 1e-9)) {
        addEdge(t.a, t.b);
        addEdge(t.b, t.c);
        addEdge(t.c, t.a);
      } else {
        keep.push(t);
      }
    }
    for (const [u, v] of edges.values()) keep.push(circum(u, v, i));
    tris = keep;
  }

  const out = [];
  for (const t of tris) {
    if (t.a < n && t.b < n && t.c < n) out.push(t.a, t.b, t.c);
  }
  return Uint16Array.from(out);
}
