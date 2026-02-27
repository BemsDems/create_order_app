import 'package:create_order_app/models/order_model.dart';

sealed class OrderState {}

class OrderInitial extends OrderState {}

class OrderLoading extends OrderState {
  final bool isRetry;
  OrderLoading({required this.isRetry});
}

class OrderSuccess extends OrderState {
  final Order order;
  OrderSuccess({required this.order});
}

class OrderError extends OrderState {
  final String message;
  OrderError(this.message);
}
