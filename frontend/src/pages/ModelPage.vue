<template>
  <div class="page-container">
    <div class="page-header">
      <h1>Model Management</h1>
      <p class="description">Manage model versions, evaluation metrics, promotion, and rollbacks.</p>
    </div>

    <div class="content-card">
      <h2>Active Model Version</h2>
      <p class="subtext">Model comparison metrics and promotion controls will connect in Phase 4.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
// Model Page Component
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
</style>
