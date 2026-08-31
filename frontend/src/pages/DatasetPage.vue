<template>
  <div class="page-container">
    <div class="page-header">
      <h1>Dataset Management</h1>
      <p class="description">Upload, validate, preview, and version datasets for model training.</p>
    </div>

    <div class="content-card">
      <h2>Upload New Dataset (.csv)</h2>
      <p class="subtext">Dataset schema validation and preview capabilities will connect in Phase 2.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
// Dataset Page Component
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
