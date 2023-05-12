import { Hello } from '../src/hello';

describe('Hello', function () {
  it('should GET / with 200 OK', function () {
    
    expect(Hello.say()).toEqual('Hello');
  });
});