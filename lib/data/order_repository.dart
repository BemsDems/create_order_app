import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:create_order_app/core/api_exception.dart';
import 'package:create_order_app/data/fake_api.dart';
import 'package:create_order_app/models/order_model.dart';
import 'package:dio/dio.dart';

class OrderRepository {
  late final Dio _dio;

  OrderRepository() {
    _dio = Dio(
      BaseOptions(
        baseUrl: 'https://api.example.com',
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 10),
      ),
    );

    _dio.interceptors.add(FakeApi());
  }

  Future<Order> createOrder({
    required int userId,
    required int serviceId,
  }) async {
    final isNetworkOffline = await Connectivity().checkConnectivity().then(
      (result) => result.contains(ConnectivityResult.none),
    );

    if (isNetworkOffline) {
      throw ApiException('Проблемы с соединением. Проверьте интернет.');
    }
    try {
      final response = await _dio.post(
        'api/orders',
        data: {'userId': userId, 'serviceId': serviceId},
      );

      if (response.statusCode == 200) {
        return Order.fromJson(response.data);
      }

      // На случай, если пришел 201 или 204, а мы ждали строго 200
      throw ApiException('Неожиданный код успеха: ${response.statusCode}');
    } on DioException catch (error) {
      if (error.response?.statusCode == 400) {
        throw ApiException('${error.response?.data['message']}');
      } else if (error.response?.statusCode == 500) {
        throw ApiException(
          'Внутренняя ошибка сервера. Пожалуйста, попробуйте позже.',
        );
      } else {
        throw ApiException('Ошибка сети');
      }
    }
  }
}
