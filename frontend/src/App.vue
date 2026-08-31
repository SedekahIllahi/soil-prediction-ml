<template>
  <DefaultLayout />
</template>

<script setup lang="ts">
import DefaultLayout from '@/layouts/DefaultLayout.vue';
</script>

<style>
/* Global reset and design system base */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  background-color: #0f172a;
  color: #f8fafc;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}
</style>
