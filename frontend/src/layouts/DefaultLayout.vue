<template>
  <div class="layout-container">
    <!-- Top Navigation Bar -->
    <header class="navbar">
      <div class="navbar-brand">
        <div class="logo-icon">🌱</div>
        <div class="brand-text">
          <span class="title">Soil Risk ML</span>
          <span class="subtitle">Road Collapse Monitoring</span>
        </div>
      </div>

      <nav class="nav-links">
        <router-link to="/" class="nav-item">Dashboard</router-link>
        <router-link to="/predict" class="nav-item">Prediction</router-link>
        <router-link to="/datasets" class="nav-item">Datasets</router-link>
        <router-link to="/models" class="nav-item">Models</router-link>
        <router-link to="/training" class="nav-item">Training</router-link>
      </nav>

      <div class="system-status">
        <span class="status-indicator"></span>
        <span class="status-text">System Ready</span>
      </div>
    </header>

    <!-- Main Content Area -->
    <main class="main-content">
      <router-view />
    </main>

    <!-- Footer -->
    <footer class="footer">
      <p>Urban Road Collapse Risk Assessment System — Academic Decision Support Prototype</p>
    </footer>
  </div>
</template>

<script setup lang="ts">
// Default Layout component setup
</script>

<style scoped>
.layout-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #0f172a;
  color: #f8fafc;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 2rem;
  background: rgba(30, 41, 59, 0.8);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  position: sticky;
  top: 0;
  z-index: 100;
}

.navbar-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.logo-icon {
  font-size: 1.5rem;
}

.brand-text {
  display: flex;
  flex-direction: column;
}

.brand-text .title {
  font-weight: 700;
  font-size: 1.1rem;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #38bdf8, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.brand-text .subtitle {
  font-size: 0.75rem;
  color: #94a3b8;
}

.nav-links {
  display: flex;
  gap: 1.5rem;
}

.nav-item {
  color: #94a3b8;
  text-decoration: none;
  font-size: 0.95rem;
  font-weight: 500;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.nav-item:hover, .nav-item.router-link-active {
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.1);
}

.system-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: #94a3b8;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #22c55e;
  box-shadow: 0 0 8px #22c55e;
}

.main-content {
  flex: 1;
  padding: 2rem;
  max-width: 1400px;
  width: 100%;
  margin: 0 auto;
  box-sizing: border-box;
}

.footer {
  text-align: center;
  padding: 1.5rem;
  font-size: 0.8rem;
  color: #64748b;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
