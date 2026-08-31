<template>
  <div class="page-container">
    <div class="page-header">
      <h1>Monitoring Dashboard</h1>
      <p class="description">Overview of urban road collapse risk predictions and active model monitoring.</p>
    </div>

    <div class="stats-grid">
      <div class="stat-card">
        <span class="stat-label">Total Predictions</span>
        <span class="stat-value">0</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Active Model</span>
        <span class="stat-value highlight">v1.0 Baseline</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">High Risk Cases</span>
        <span class="stat-value warning">0</span>
      </div>
    </div>

    <div class="content-card placeholder-card">
      <h2>Recent Risk Activity</h2>
      <p>No prediction activity logged yet. Phase 5 will connect live prediction storage.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
// Dashboard Page Component
</script>

<style scoped>
.page-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.page-header h1 {
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
}

.description {
  color: #94a3b8;
  font-size: 0.95rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
}

.stat-card {
  background: #1e293b;
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 1.25rem;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.stat-label {
  font-size: 0.85rem;
  color: #94a3b8;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
}

.stat-value.highlight {
  color: #38bdf8;
}

.stat-value.warning {
  color: #f87171;
}

.content-card {
  background: #1e293b;
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 1.5rem;
  border-radius: 10px;
}

.placeholder-card p {
  color: #94a3b8;
  margin-top: 0.5rem;
}
</style>
