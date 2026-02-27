import 'package:create_order_app/core/api_exception.dart';
import 'package:create_order_app/data/order_repository.dart';
import 'package:create_order_app/presentation/main/controller/order_state.dart';
import 'package:flutter/material.dart';

class OrderController {
  final orderRepository = OrderRepository();
  ValueNotifier<OrderState> statusNotifier = ValueNotifier(OrderInitial());

  Future<void> submitOrder({
    required int userId,
    required int serviceId,
  }) async {

    final isRetry = statusNotifier.value is OrderError;

    statusNotifier.value = OrderLoading(isRetry: isRetry);

    try {
      final order = await orderRepository.createOrder(
        userId: userId,
        serviceId: serviceId,
      );

      statusNotifier.value = OrderSuccess(order: order);
    } catch (error) {
      if (error is ApiException) {
        statusNotifier.value = OrderError(
          'Не удалось создать заказ: ${error.message}',
        );
      } else {
        statusNotifier.value = OrderError(
          'Ошибка при создании заказа: ${error.toString()}',
        );
      }
    }
  }
}
