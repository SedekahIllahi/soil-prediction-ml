<template>
  <div class="page-container">
    <div class="page-header">
      <h1>Road Collapse Risk Prediction</h1>
      <p class="description">Input road segment measurements to calculate soil risk classification.</p>
    </div>

    <div class="content-card">
      <h2>Feature Inputs</h2>
      <p class="subtext">Canonical feature schema (34 features) will be rendered dynamically in Phase 5.</p>
      <div class="action-bar">
        <button class="primary-btn" disabled>Calculate Risk (Phase 5)</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Prediction Page Component
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

.description, .subtext {
  color: #94a3b8;
  font-size: 0.95rem;
}

.content-card {
  background: #1e293b;
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 1.5rem;
  border-radius: 10px;
}

.action-bar {
  margin-top: 1.5rem;
}

.primary-btn {
  background: #38bdf8;
  color: #0f172a;
  border: none;
  padding: 0.6rem 1.2rem;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  opacity: 0.6;
}
</style>
