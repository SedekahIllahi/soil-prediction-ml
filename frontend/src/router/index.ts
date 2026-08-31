import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';
import DashboardPage from '@/pages/DashboardPage.vue';
import PredictionPage from '@/pages/PredictionPage.vue';
import DatasetPage from '@/pages/DatasetPage.vue';
import ModelPage from '@/pages/ModelPage.vue';
import TrainingPage from '@/pages/TrainingPage.vue';

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Dashboard',
    component: DashboardPage,
    meta: { title: 'Dashboard' }
  },
  {
    path: '/predict',
    name: 'Prediction',
    component: PredictionPage,
    meta: { title: 'Risk Prediction' }
  },
  {
    path: '/datasets',
    name: 'Datasets',
    component: DatasetPage,
    meta: { title: 'Dataset Management' }
  },
  {
    path: '/models',
    name: 'Models',
    component: ModelPage,
    meta: { title: 'Model Management' }
  },
  {
    path: '/training',
    name: 'Training',
    component: TrainingPage,
    meta: { title: 'ML Training' }
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach((to, _from, next) => {
  document.title = `${String(to.meta.title || 'App')} - Soil ML Prediction System`;
  next();
});

export default router;
