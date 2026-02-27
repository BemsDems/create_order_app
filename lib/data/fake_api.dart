import 'dart:math';

import 'package:dio/dio.dart';

class FakeApi extends Interceptor {
  @override
  void onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    await Future.delayed(const Duration(seconds: 2));

    if (options.path.contains('/orders')) {
      final data = options.data as Map<String, dynamic>?;

      final userId = data?['userId'];
      final serviceId = data?['serviceId'];

      if (userId == null || serviceId == null) {
        final errorMessage =
            'Отсутствуют обязательные поля: ${userId == null ? 'userId' : ''} ${serviceId == null ? 'serviceId' : ''}';
        return handler.reject(
          DioException(
            requestOptions: options,
            response: Response(
              requestOptions: options,
              statusCode: 400,
              data: {'message': errorMessage},
            ),
          ),
        );
      }

      final random = Random();
      final randomOrderId = random.nextInt(90000) + 10000;

      final statuses = ['pending', 'created', 'processing'];
      final randomStatus = statuses[random.nextInt(statuses.length)];

      final responseData = {
        "order_id": randomOrderId,
        "status": randomStatus,
        "payment_url": "https://fake-pay.com/order/$randomOrderId",
      };

      return handler.resolve(
        Response(requestOptions: options, data: responseData, statusCode: 200),
      );
    }

    super.onRequest(options, handler);
  }
}
