import 'package:create_order_app/presentation/main/controller/order_controller.dart';
import 'package:create_order_app/presentation/main/controller/order_state.dart';
import 'package:flutter/material.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final OrderController _controller = OrderController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Главный экран')),
      body: ValueListenableBuilder(
        valueListenable: _controller.statusNotifier,
        builder: (context, state, child) {
          final text = switch (state) {
            OrderInitial() =>
              'Добро пожаловать в приложение для создания заказов!',
            OrderSuccess() => 'Заказ ${state.order.orderId} успешно создан!',
            OrderError() => state.message,
            OrderLoading() => '',
          };

          final buttonText = switch (state) {
            OrderError() => 'Повторить попытку',
            OrderLoading() =>
              state.isRetry ? 'Повторить попытку' : 'Создать заказ',
            _ => 'Создать заказ',
          };

          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                if (state is OrderLoading) const CircularProgressIndicator(),
                Text(
                  text,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: state is OrderLoading
                      ? null
                      : () => _controller.submitOrder(userId: 1, serviceId: 1),
                  child: Text(buttonText),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}
